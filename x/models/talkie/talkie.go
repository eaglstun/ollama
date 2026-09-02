// Package talkie provides the custom "talkie" 1930-era decoder-only transformer
// for the MLX runner. It is a GPT with RoPE, Q/K RMS normalisation, SwiGLU MLPs,
// per-head / per-layer / lm-head learned gains, and an embedding-skip connection.
//
// It deliberately mirrors the reference implementations
//   - PyTorch:  ~/Documents/AI/talkie/src/talkie/model.py
//   - MLX:      ~/Documents/AI/talkie/src/talkie/mlx/model.py
//
// and loads the original checkpoint tensor names unchanged (embed.weight,
// blocks.N.attn.attn_query.weight, blocks.N.attn.head_gain.head_g, etc.).
//
// Two details differ from a stock Llama and are load-bearing:
//   - RMSNorm is WEIGHTLESS and computed in fp32 with float32 machine epsilon
//     (no learned gamma; there are no *norm.weight tensors in the checkpoint).
//   - RoPE uses the opposite rotation sign from standard NeoX:
//       y1 = x1*cos + x2*sin ; y2 = -x1*sin + x2*cos
//     so we apply rotation by hand rather than via mlx.RoPEWithBase.
package talkie

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("TalkieForCausalLM", newModel)
}

// rmsEps matches torch.nn.functional.rms_norm with eps=None for fp32 inputs:
// the float32 machine epsilon (2^-23). The reference computes the reduction in
// fp32, so we do too.
const rmsEps float32 = 1.1920929e-07

// Config holds talkie model configuration. Field names match talkie's native
// config.json (n_layer, n_head, ...); "architectures" is read separately by the
// runner's base.New to route here.
type Config struct {
	NLayer    int32   `json:"n_layer"`
	NHead     int32   `json:"n_head"`
	NEmbd     int32   `json:"n_embd"`
	HeadDim   int32   `json:"head_dim"`
	VocabSize int32   `json:"vocab_size"`
	RopeBase  float32 `json:"rope_base"`
	MaxSeqLen int32   `json:"max_seq_len"`

	// Quantization parameters (resolved during load). talkie ships bf16 today,
	// so these are typically zero/empty, but the plumbing matches other archs.
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed.
	Scale float32 `json:"-"`
}

// Model is the talkie text model.
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	LMHead      *mlx.Array // raw [vocab, n_embd]; untied, not a Linear
	LMHeadGain  *mlx.Array // scalar [1]

	// Precomputed RoPE tables, shape [max_seq_len, head_dim/2].
	CosTable *mlx.Array
	SinTable *mlx.Array

	tok *tokenizer.Tokenizer
	*Config
}

type Layer struct {
	Attn *Attention
	MLP  *MLP

	AttnGain  *mlx.Array // [1]
	MLPGain   *mlx.Array // [1]
	EmbedSkip *mlx.Array // [1]
}

type Attention struct {
	Q, K, V, Resid nn.LinearLayer
	HeadGain       *mlx.Array // [n_head], per-head gain on the query
}

type MLP struct {
	Gate, Linear, Resid nn.LinearLayer
}

func newModel(root *model.Root) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if cfg.NEmbd <= 0 || cfg.NHead <= 0 || cfg.NLayer <= 0 {
		return nil, fmt.Errorf("invalid config: n_embd=%d n_head=%d n_layer=%d", cfg.NEmbd, cfg.NHead, cfg.NLayer)
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = cfg.NEmbd / cfg.NHead
	}
	if cfg.NHead*cfg.HeadDim != cfg.NEmbd {
		return nil, fmt.Errorf("n_head (%d) * head_dim (%d) != n_embd (%d)", cfg.NHead, cfg.HeadDim, cfg.NEmbd)
	}
	if cfg.HeadDim%2 != 0 {
		return nil, fmt.Errorf("head_dim (%d) must be even for RoPE", cfg.HeadDim)
	}
	if cfg.RopeBase == 0 {
		cfg.RopeBase = 1_000_000
	}
	if cfg.MaxSeqLen == 0 {
		cfg.MaxSeqLen = 2048
	}
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	if qt := root.QuantType(); qt != "" {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams(qt)
		if gs := root.GroupSize(); gs > 0 {
			cfg.QuantGroupSize = gs
		}
	} else {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams("")
	}
	cfg.TensorQuant = root.AllTensorQuant()

	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	tokConfig := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	if d, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = d
	}
	if d, err := root.Manifest.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = d
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	return &Model{
		Layers: make([]*Layer, cfg.NLayer),
		Config: &cfg,
		tok:    tok,
	}, nil
}

// LoadWeights wires the checkpoint tensors into the model. The RoPE tables are
// built here too so base.Weights pins them alongside the loaded weights.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	if m.EmbedTokens = model.MakeEmbeddingLayer(tensors, "embed", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant); m.EmbedTokens == nil {
		return fmt.Errorf("missing embed.weight")
	}

	if m.LMHead = tensors["lm_head"]; m.LMHead == nil {
		return fmt.Errorf("missing lm_head")
	}
	if m.LMHeadGain = tensors["lm_head_gain.w_g"]; m.LMHeadGain == nil {
		return fmt.Errorf("missing lm_head_gain.w_g")
	}

	for i := range m.NLayer {
		p := fmt.Sprintf("blocks.%d", i)
		l := &Layer{Attn: &Attention{}, MLP: &MLP{}}

		l.Attn.Q = linears.Make(p + ".attn.attn_query")
		l.Attn.K = linears.Make(p + ".attn.attn_key")
		l.Attn.V = linears.Make(p + ".attn.attn_value")
		l.Attn.Resid = linears.Make(p + ".attn.attn_resid")
		l.Attn.HeadGain = tensors[p+".attn.head_gain.head_g"]

		l.MLP.Gate = linears.Make(p + ".mlp.mlp_gate")
		l.MLP.Linear = linears.Make(p + ".mlp.mlp_linear")
		l.MLP.Resid = linears.Make(p + ".mlp.mlp_resid")

		l.AttnGain = tensors[p+".attn_gain.a_g"]
		l.MLPGain = tensors[p+".mlp_gain.a_g"]
		l.EmbedSkip = tensors[p+".embed_skip.a_g"]

		if l.Attn.Q == nil || l.Attn.K == nil || l.Attn.V == nil || l.Attn.Resid == nil {
			return fmt.Errorf("layer %d: missing attention projection", i)
		}
		if l.MLP.Gate == nil || l.MLP.Linear == nil || l.MLP.Resid == nil {
			return fmt.Errorf("layer %d: missing mlp projection", i)
		}
		if l.Attn.HeadGain == nil || l.AttnGain == nil || l.MLPGain == nil || l.EmbedSkip == nil {
			return fmt.Errorf("layer %d: missing gain tensor", i)
		}
		m.Layers[i] = l
	}

	m.buildRoPE()
	return nil
}

// buildRoPE precomputes cos/sin tables [max_seq_len, head_dim/2] in bf16, with
// inv_freq = base^(-2i/head_dim), matching the reference precompute.
func (m *Model) buildRoPE() {
	half := int(m.HeadDim / 2)
	T := int(m.MaxSeqLen)
	cos := make([]float32, T*half)
	sin := make([]float32, T*half)
	for t := 0; t < T; t++ {
		for i := 0; i < half; i++ {
			invFreq := math.Pow(float64(m.RopeBase), -float64(2*i)/float64(m.HeadDim))
			ang := float64(t) * invFreq
			cos[t*half+i] = float32(math.Cos(ang))
			sin[t*half+i] = float32(math.Sin(ang))
		}
	}
	m.CosTable = mlx.FromValues(cos, T, half).AsType(mlx.DTypeBFloat16)
	m.SinTable = mlx.FromValues(sin, T, half).AsType(mlx.DTypeBFloat16)
}

// rmsNorm is weightless RMS normalisation over the last axis, computed in fp32
// and cast back to bf16 (matches talkie's F.rms_norm with no weight).
func rmsNorm(x *mlx.Array) *mlx.Array {
	axis := len(x.Dims()) - 1
	xf := x.AsType(mlx.DTypeFloat32)
	ms := mlx.Mean(mlx.Mul(xf, xf), axis, true)
	denom := mlx.Add(ms, mlx.NewScalarArray(rmsEps)).Sqrt()
	return mlx.Div(xf, denom).AsType(mlx.DTypeBFloat16)
}

// rope applies talkie's half-rotation RoPE (negative-sign convention) to a
// [B, L, H, D] tensor, gathering cos/sin by per-token absolute position.
func (m *Model) rope(x, posIdx *mlx.Array, B, L int32) *mlx.Array {
	half := int(m.HeadDim / 2)
	cos := mlx.Reshape(mlx.Take(m.CosTable, posIdx, 0), B, L, 1, int32(half))
	sin := mlx.Reshape(mlx.Take(m.SinTable, posIdx, 0), B, L, 1, int32(half))

	x1 := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, half))
	x2 := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(half, mlx.End))

	y1 := mlx.Add(mlx.Mul(x1, cos), mlx.Mul(x2, sin))
	y2 := mlx.Add(mlx.MulScalar(mlx.Mul(x1, sin), -1), mlx.Mul(x2, cos))
	return mlx.Concatenate([]*mlx.Array{y1, y2}, 3)
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	// Per-token absolute positions: row b starts at SeqOffsets[b].
	pos := make([]int32, int(B)*int(L))
	for bi := 0; bi < int(B); bi++ {
		off := b.SeqOffsets[bi]
		for t := 0; t < int(L); t++ {
			pos[bi*int(L)+t] = off + int32(t)
		}
	}
	posIdx := mlx.FromValues(pos, int(B)*int(L))

	h := rmsNorm(m.EmbedTokens.Forward(b.InputIDs))
	eX := h
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(m, h, eX, b, c, posIdx, B, L)
	}

	// talkie has no draft/multi-token-prediction head, so the auxiliary hidden
	// state is just the final hidden state, as in the other dense archs.
	out := rmsNorm(h)
	return out, out
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	w := mlx.Mul(m.LMHead, m.LMHeadGain) // [vocab, n_embd]
	return mlx.Matmul(x, w.Transpose(1, 0))
}

func (m *Model) NumLayers() int { return len(m.Layers) }

func (m *Model) MaxContextLength() int { return int(m.MaxSeqLen) }

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = cache.NewKVCache()
	}
	return caches
}

func (l *Layer) Forward(m *Model, x, eX *mlx.Array, b *batch.Batch, c cache.Cache, posIdx *mlx.Array, B, L int32) *mlx.Array {
	attn := l.Attn.Forward(m, rmsNorm(x), b, c, posIdx, B, L)
	x = mlx.Add(x, mlx.Mul(attn, l.AttnGain))
	mlpOut := l.MLP.Forward(rmsNorm(x))
	x = mlx.Add(x, mlx.Mul(mlpOut, l.MLPGain))
	x = mlx.Add(x, mlx.Mul(eX, l.EmbedSkip))
	return x
}

func (a *Attention) Forward(m *Model, x *mlx.Array, b *batch.Batch, c cache.Cache, posIdx *mlx.Array, B, L int32) *mlx.Array {
	q := mlx.Reshape(a.Q.Forward(x), B, L, m.NHead, m.HeadDim)
	k := mlx.Reshape(a.K.Forward(x), B, L, m.NHead, m.HeadDim)
	v := mlx.Reshape(a.V.Forward(x), B, L, m.NHead, m.HeadDim)

	// Order matches the reference: RoPE, then Q/K RMSNorm, then per-head gain.
	q = rmsNorm(m.rope(q, posIdx, B, L))
	k = rmsNorm(m.rope(k, posIdx, B, L))
	q = mlx.Mul(q, mlx.Reshape(a.HeadGain, 1, 1, m.NHead, 1))

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	var kv nn.SDPAOption
	if c != nil {
		kv = nn.WithKVHistory(c.(cache.Attention).Update(b, k, v))
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
	}
	out := nn.ScaledDotProductAttention(b, q, m.Scale, kv, nn.WithMask(nn.CausalMask()))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, m.NEmbd)
	return a.Resid.Forward(out)
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	// silu(gate) * linear, from primitives (mirrors the reference; avoids the
	// fused compiled SwiGLU kernel).
	gate := m.Gate.Forward(x)
	silu := mlx.Mul(gate, mlx.Sigmoid(gate))
	return m.Resid.Forward(mlx.Mul(silu, m.Linear.Forward(x)))
}
