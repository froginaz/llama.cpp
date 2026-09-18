# Sequence: GGUF + per-architecture C++ ⇒ ggml_cgraph

A single sequence diagram tracing the whole progression: the **GGUF file** supplies *weights and dimensions* (data), the **per-architecture C++ builder** (`src/models/llama.cpp`) supplies the *op wiring* (control), and `ggml_build_forward_expand` fuses them into one ordered `ggml_cgraph`. All call sites are cited to source. See [04-graph-construction.md](04-graph-construction.md) and [05-ggml-cgraph.md](05-ggml-cgraph.md) for the prose.

## Full sequence (load phase + build phase)

```plantuml
@startuml
title GGUF + per-architecture C++ builder  =>  ggml_cgraph
skinparam sequenceMessageAlign center
skinparam maxMessageSize 260

actor User
participant "GGUF file\n(model.gguf)" as GGUF
participant "llama_model_loader\n(llama-model-loader.cpp)" as Loader
participant "arch registry\n(llama-arch.cpp)" as Arch
participant "llama_model\n(llama-model.cpp)" as Model
participant "per-arch builder\n(src/models/llama.cpp)" as Builder
participant "llm_graph_context\n(llama-graph.cpp)" as GC
participant "ggml ctx0\n(ggml.c)" as GGML
participant "ggml_cgraph gf" as GF

== Load phase (once per model) ==
User -> Loader : llama_model_load_from_file(path)
activate Loader
Loader -> GGUF : gguf_init_from_reader()  (gguf.cpp:451)
GGUF --> Loader : KV metadata + tensor-info index
Loader -> Arch : llm_arch_from_string("llama")  (llama-arch.cpp:830)
Arch --> Loader : LLM_ARCH_LLAMA  (selects model subclass)

Loader -> Model : load_hparams()  (llama-model.cpp:1017)
activate Model
Model -> Arch : LLM_KV(llama.attention.head_count)
Arch --> Model : concrete metadata key
Model -> GGUF : get_key(key)
GGUF --> Model : value
note right of Model : fills llama_hparams\n(n_layer, n_head, n_embd_head, n_rot)

Model -> Model : load_tensors()  (llama-model.cpp:1203)
loop each weight, per layer il
  Model -> Arch : LLM_TN tn(ATTN_Q, il)  ->  "blk.il.attn_q.weight"
  Arch --> Model : tensor name
  Model -> Loader : create_tensor(name)  (llama-model-loader.cpp:1046)
  Loader -> GGUF : tensor data (mmap)
  Loader --> Model : ggml_tensor*  (bound to backend buffer)
end
note right of Model : weights now live as\nggml_tensor* on llama_layer
deactivate Model
Loader --> User : llama_model*
deactivate Loader

== Build phase (every llama_decode / ubatch) ==
User -> Model : build_graph(params)  (llama-model.cpp:2186)
activate Model
Model -> Builder : build_arch_graph(params)  (src/models/llama.cpp:94)
activate Builder
Builder -> Builder : graph<false>::graph(model, params)  (src/models/llama.cpp:99)

Builder -> GC : build_inp_embd(model.tok_embd)  (llama-graph.cpp:1793)
GC -> GGML : ggml_get_rows(tok_embd, tokens)
GGML --> GC : inpL [n_embd, n_tokens]
GC --> Builder : inpL

loop il = 0 .. n_layer-1
  Builder -> GC : build_norm(inpL, attn_norm, RMS)  (llama-graph.cpp:1117)
  GC -> GGML : ggml_rms_norm  then  ggml_mul
  Builder -> GC : build_qkv(layer, cur)  (llama-graph.cpp:1153)
  GC -> GGML : ggml_mul_mat(wq), ggml_mul_mat(wk), ggml_mul_mat(wv)
  Builder -> GGML : ggml_rope_ext(Qcur, Kcur, inp_pos)
  Builder -> GC : build_attn(...)  (llama-graph.cpp:2271)
  GC -> GGML : cpy_k/cpy_v to KV cache; ggml_mul_mat(K,Q); ggml_soft_max_ext; ggml_mul_mat(V,kq); ggml_mul_mat(wo)
  Builder -> GGML : ggml_add  (attention residual)
  Builder -> GC : build_norm(ffn_inp, ffn_norm, RMS)
  Builder -> GC : build_ffn(... SwiGLU ...)  (llama-graph.cpp:1230)
  GC -> GGML : ggml_mul_mat(up), ggml_mul_mat(gate), ggml_swiglu_split, ggml_mul_mat(down)
  Builder -> GGML : ggml_add  (FFN residual)
end
note over GGML : every ggml_* call allocates a result tensor\nwith op + src[] in ctx0 — no math runs yet

Builder -> GC : build_norm(cur, output_norm, RMS)
Builder -> GC : build_lora_mm(model.output, cur)   (lm_head)
GC -> GGML : ggml_mul_mat(output)  ->  logits [n_vocab, n_tokens]
Builder --> Model : llm_graph_context (res->t_logits = cur)
deactivate Builder

Model -> GF : ggml_build_forward_expand(gf, t_logits)  (ggml.c:6959)
activate GF
GF -> GGML : ggml_visit_parents(): DFS over src[] from logits
GGML --> GF : append ops to nodes[] in topological order
GF --> Model : ggml_cgraph ready  (nodes[] = execution schedule)
deactivate GF
Model --> User : ggml_cgraph* gf
deactivate Model

note over GGUF, GF
  GGUF supplies WEIGHTS + DIMENSIONS (data).
  The per-arch C++ builder supplies the OP WIRING (control).
  ggml_build_forward_expand fuses them into one ordered cgraph.
end note
@enduml
```

## The essence in one picture

```plantuml
@startuml
title data + control  =>  cgraph
left to right direction

rectangle "GGUF file" as G #e3f2fd {
  card "hparams\n(n_layer, n_head, ...)" as H
  card "weights\n(blk.N.attn_q.weight, ...)" as W
}
rectangle "per-arch C++\n(src/models/llama.cpp)" as C #fff3e0 {
  card "op wiring\nnorm -> qkv -> rope ->\nattn -> ffn -> residual" as O
}
rectangle "ggml_build_forward_expand" as E #ede7f6
rectangle "ggml_cgraph\n(ordered nodes[])" as R #e8f5e9

H --> O : sizes every op
W --> O : feeds ggml_mul_mat
O --> E : logits expression tree
E --> R : topological sort
@enduml
```

## How to read it

- **Load phase** runs once: the loader parses GGUF, the `general.architecture` key picks `LLM_ARCH_LLAMA`, `load_hparams` reads dimensions through `LLM_KV` keys, and `load_tensors` binds each GGUF tensor (named via `LLM_TN`) to a `ggml_tensor*` field on `llama_layer`.
- **Build phase** runs every `llama_decode`: the per-architecture builder walks `n_layer` blocks, calling `llm_graph_context` helpers that chain `ggml_*` ops over the loaded weights. Each `ggml_*` call only *allocates* a tagged tensor (`op` + `src[]`) in `ctx0` — no arithmetic happens.
- **The fuse point** is `ggml_build_forward_expand(gf, t_logits)`: it depth-first walks `src[]` back from the logits tensor and appends every reachable op to `gf->nodes[]` in topological (execution) order. That node array *is* the computation graph.

> Mental model: **GGUF = data, per-arch C++ = control, `ggml_build_forward_expand` = the combiner** that turns the expression tree into an ordered `ggml_cgraph`.

See [03-hparams-and-tensors.md](03-hparams-and-tensors.md) for the load phase, [04-graph-construction.md](04-graph-construction.md) for the build phase, and [06-execution-and-scheduling.md](06-execution-and-scheduling.md) for what happens to the finished graph.
