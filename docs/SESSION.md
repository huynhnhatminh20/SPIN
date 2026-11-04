# SESSION — SPIN × Qwen3-0.6B × GSM8K-VI

Cập nhật file này sau mỗi bước xong. Quy ước: `[x]` xong, `[ ]` chưa, `[-]` đang làm.

## Mục tiêu
`Qwen/Qwen3-0.6B → SFT → SPIN-1 → SPIN-2 → Evaluation` | data train 4000 / val 500 / test 500 | `k=1` | A100 40GB ×1 | **budget $5 / session (đã nạp)**.
Mở rộng (nick mới `huynhnhatminh20102005`, Starter $30): `SPIN-2 → SPIN-3 → SPIN-4 → Evaluation` (gen từ ckpt trước, cùng config/beta, `k=1`).

## Trạng thái hiện tại (22/09/2026)
- [x] Viết 8 file wrapper trong `Downloads\SPIN` (= `/src/app` trên Modal)
- [x] `py_compile` pass 8/8 file `.py`, `config_spin_a100.yaml` parse OK
- [x] Guard "không duplicate SPIN" pass (không có `spin_loss`/`SPINTrainer` trong wrapper)
- [x] Viết `modal_app.py` (image clone official + Volume `spin-artifacts` + A100 40GB + timeout 3h)
- [x] Fix `convert_qwen3.py`: dùng `save_to_disk` (Arrow dir) thay vì parquet lẻ — khớp `load_from_disk` của official `data.py`
- [x] Clone official `uclaml/SPIN` → `/src/official_spin` (xảy ra khi build Modal image)
- [x] Chạy `preflight` trên Modal (CPU, $0 GPU) — PASS
- [x] Session 1 — DONE (238/238 steps, loss 0.683, `spin1/checkpoint-238`)
- [x] Session 2 — DONE (gen2 4000 → spin2 238/238 steps loss 0.684 → eval 4 models)
- [x] Tổng kết kết quả base 6% / sft 33.6% / spin1 32.8% / spin2 34.8% (500 test)

## Checklist artifacts (đánh dấu khi có file thật)
- [x] `artifacts/data/{train,validation,test}.jsonl` (4000/500/500) + `split_metadata.json`
- [x] `artifacts/data/sft_{train,validation}.jsonl`
- [x] `artifacts/sft/` (checkpoint SFT)
- [x] `artifacts/spin1_generation/gen_iter0.jsonl` (4000 dòng, k=1)
- [x] `artifacts/spin1_data/{train,test}/` (parquet official format)
- [x] `artifacts/spin1/` (checkpoint SPIN-1) + `run_metadata_session1.json` (có `official_spin_commit`)
- [x] `artifacts/spin2_generation/gen_iter1.jsonl`
- [x] `artifacts/spin2_data/{train,test}/`
- [x] `artifacts/spin2/` (checkpoint SPIN-2) + `run_metadata_session2.json`
- [x] `artifacts/evaluation/summary.json` (4 con số accuracy)
- [x] `artifacts/spin3_generation/gen_iter2.jsonl` (4000 dòng, k=1)
- [x] `artifacts/spin3_data/{train,test}/`
- [x] `artifacts/spin3/` (checkpoint SPIN-3, `checkpoint-238`, loss 0.6854) + `run_metadata_session3.json`
- [x] `artifacts/evaluation/summary.json` cập nhật 5 models (merge mode giữ số cũ)
- [x] `artifacts/spin4_generation/gen_iter3.jsonl` (4000 dòng, k=1)
- [x] `artifacts/spin4_data/{train,test}/`
- [x] `artifacts/spin4/` (checkpoint SPIN-4, `checkpoint-238`, loss 0.6875) + `run_metadata_session4.json`
- [x] `artifacts/evaluation/summary.json` FINAL 6 models: base 6.0 / sft 33.6 / spin1 32.8 / spin2 34.8 / spin3 34.2 / spin4 32.6

## Mở rộng SPIN-3/4 (nick mới, 23/09/2026) — CODE XONG, chờ GPU
- [x] `modal_app.py`: thêm `session3()` + `session4()` (A100 40GB, timeout **3h** — tăng từ 2h vì eval 5-6 model)
- [x] `app.py`: thêm `stage_session3()` (gen `spin2`→`gen_iter2`→`spin3_data`→SPIN-3→eval 5 model) + `stage_session4()` (tương tự, eval 6 model); helper `_summary_has()` để skip eval khi resume (tiết kiệm ~1h GPU)
- [x] `app.py --stage` choices + `launch_detached.py` allowlist/help cập nhật `session3/session4`
- [x] Verify: `py_compile` 10/10 pass, `config_spin_a100.yaml` parse OK, guard duplicate-SPIN pass, `_summary_has` unit 5/5 case pass
- [x] Migrate Volume sang nick mới: `spin-artifacts` đủ 12/12 mục (data 4000/500/500, sft/spin1/spin2 ckpt header OK 310 tensors, gen 4000/4000, Arrow data OK, 0 file 0-byte); image build OK (116s)
- [ ] GPU nick mới: Modal chặn `deploy`/`run` — `Please add a payment method to use A100-40GB` (Starter $30 còn nguyên $0 spend). User đang xử lý với Modal. Sau khi mở cổng: `deploy` → `status` → `session3` → `session4`.

## Lệnh chạy (trên Modal, theo thứ tự)
```bash
modal run modal_app.py::preflight
modal run modal_app.py::session1    # xong → THOÁT, release A100
modal run modal_app.py::session2    # xong → THOÁT, release A100
```

## Ràng buộc không được quên
- `k=1`, không đổi thành k=4. `enable_thinking=False`. Không sửa `trainer.py::spin_loss`.
- Official clone ở `/src/official_spin`, wrapper ở `/src/app` — không trộn.
- Mỗi session xong phải terminate process để release GPU (cost safety).
- **Budget $5/session**: preflight chạy CPU ($0); SFT 3 epochs (train rẻ); gen/eval `max_new_tokens=256`, batch 16 (đáp án GSM8K ngắn — lựa chọn kỹ thuật, không phải cắt giảm); timeout mỗi session 2h. Ước tính: S1 ~2–2.5h ≈ $3–3.8, S2 ~1.5–2h ≈ $2.3–3 — mỗi session vừa dưới $5.

## Nhật ký
- 22/09/2026: khởi tạo project, viết xong wrapper + kiểm tra tĩnh. Việc tiếp theo: setup Modal image + chạy preflight.
- 22/09/2026: đọc official `convert_data.py` + `finetune.sh` → fix converter sang `save_to_disk`, viết `modal_app.py`. Việc tiếp theo: `modal run modal_app.py::preflight`.
- 22/09/2026: **preflight PASS** (Modal CPU). official_spin_commit=`a12ba80`, transformers 5.17 + Qwen3 template OK, converter 8/8 valid. Việc tiếp theo: `modal run modal_app.py::session1` (A100, ~$3–3.8).
- 22/09/2026: session1 chạy 3 lần fail-nhanh (fix: assert `<think>` rỗng thay vì vắng mặt; `max_length` thay `max_seq_length` cho trl 1.13) rồi chạy thật tới SFT ~12% thì user abort client → job dừng, Volume chưa commit, không mất tiền đáng kể. Mở lại chat: kiểm tra `modal app list` rồi chạy lại session1.
- 22/09/2026: chuyển sang kiến trúc detached (`modal deploy` + `Function.spawn` qua `launch_detached.py`, thêm hàm `status`). Session1 đã spawn detached, đang chạy độc lập trên cloud.
- 22/09/2026: session1 FAIL 16:40 ở `accelerate launch run_spin.py`: `ImportError is_peft_available` (official SPIN đòi trl~=0.7, image cài trl 1.13). SFT+gen+convert vẫn xong, Volume còn đủ data. Fix option 1: thêm `trl_compat_shim.py` (vá 4 tên thiếu: reuse real `is_peft_available`/`create_reference_model`/`disable_dropout`, fallback `is_wandb_available`/`PreTrainedModelWrapper`/`pad_to_length`; không đụng spin_loss) + `app.py` chạy official qua launcher pre-import shim + resume-by-ckpt (skip split/reformat/sft/gen/convert/spin khi outputs đã có; session2 tương tự + skip eval) + `config_spin_a100.yaml`: `evaluation_strategy` → `eval_strategy` (transformers 5.x đã xóa tên cũ). Verify: py_compile 10/10 pass, 3 dòng import official pass local, guard duplicate-SPIN pass. Việc tiếp theo: deploy lại + spawn lại session1 (sẽ skip tới SPIN-1).
- 22/09/2026: respawn session1 (`fc-01M348Z3WJVWN3AJWPFVS3VEWM`) — shim OK (import official pass) nhưng FAIL tiếp 17:01 ở parse yaml: `warmup_ratio` cũng bị transformers 5.x xóa (chỉ còn `warmup_steps`). Fix: `warmup_steps: 24` (≈ ratio 0.1 trên ~238 steps). Deploy + respawn lần 3 (`fc-01M3493PCA9KBNN55FEEYWVV3A`), resume skip tới SPIN-1.
- 22/09/2026: lần 3 tiến sâu (yaml parse OK, format data 200/200 test, load config+weights SFT) rồi FAIL 17:04 ở `from_pretrained`: official truyền `use_flash_attention_2`, key này transformers 5.x đã xóa → `TypeError` trong `Qwen3ForCausalLM.__init__`. Fix: shim bọc `AutoModelForCausalLM.from_pretrained` tự pop key (True → map sang `attn_implementation`), verify local pass. Deploy + respawn lần 4 (`fc-01M349MZ9AQF0Y4HN8QV7R7QYX`).
- 22/09/2026: lần 4 FAIL nhanh 17:14 ở `super().__init__` của official SPINTrainer: transformers 5.x xóa kwarg `tokenizer` (đổi thành `processing_class`). Chưa fix — DỪNG theo yêu cầu, chờ lệnh mới. GPU đã release (process exit), không tốn tiền thêm.
- 22/09/2026: user lệnh fix tiếp. Vá 2 chỗ (không đụng official): (1) shim bọc `Trainer.__init__` map `tokenizer`→`processing_class` + thêm property `Trainer.tokenizer` (5.x đã xóa attr này; official đọc `self.tokenizer` khi `generate_during_eval`); verify local trên Trainer thật: mapping + property đều pass. (2) launcher post-import patch `SPINTrainer.compute_loss` nuốt `num_items_in_batch` (5.x `training_step` luôn truyền kwarg này — đã grep source 5.15.1 xác nhận, official thiếu là crash step đầu). Deploy + respawn lần 5 (`fc-01M34ASYDE5XNYC6ASBM96V3F0`).
- 22/09/2026: lần 5 FAIL 17:33 (đã qua Trainer init): `AcceleratorState ... mixed_precision='bf16'` — official gọi `Accelerator()` trần (default `no`), Trainer đòi `bf16` → xung đột state. Fix: `accelerate launch --mixed_precision=bf16` (khớp yaml `bf16:true`, không đụng official). Deploy + respawn lần 6 (`fc-01M34B1E5BFWW8SB4CJB86PG56`).
- 22/09/2026: lần 6 FAIL 17:38 ở step train đầu (đã qua init/model-load/format 3800+200): `TypeError: SPINTrainer.get_batch_samples() takes 3 positional arguments but 4 were given` — official định nghĩa `get_batch_samples(self, model, batch)` để generate mẫu, trùng TÊN với `Trainer.get_batch_samples(self, epoch_iterator, num_batches, device)` mới của transformers 5.x (4.37 chưa có hàm này nên không lộ). Volume giữ đủ `data/ sft/ spin1_generation/ spin1_data/`, `spin1/` vẫn rỗng. Fix: launcher dispatch theo signature (batch-dict → SPIN-generate, epoch_iterator+num_batches → base Trainer; official untouched), verify local mock pass + py_compile + guard pass. Deploy OK + respawn lần 7 (`fc-01M34KBN2RCA85ZDX5AXJK4764`), resume skip tới SPIN-1.
- 22/09/2026: lần 7 FAIL 20:03 nhưng TIẾN SÂU (dispatch `get_batch_samples` OK, train chạy 1/238 step ~15s, loss 0.6931): `TypeError: SPINTrainer.log() takes 2 positional arguments but 3 were given` ở `_maybe_log_save_evaluate → self.log(logs, start_time)` — official `log(self, logs)` thiếu `start_time` của transformers 5.x. Fix: launcher bọc `SPINTrainer.log` nuốt `*args/**kwargs` (official untouched). Deploy OK + respawn lần 8 (`fc-01M34KNZRFRD2RY8GZ3AHQEZAY`), resume skip tới SPIN-1.
- 22/09/2026: lần 8 **SESSION1 DONE** (238/238 steps, ~47 phút train; qua epoch-1 save rồi chạy tiếp epoch-2 tới `checkpoint-238` + `train_results.json`). Volume có đủ `spin1/{checkpoint-238,model.safetensors,config.json,trainer_state.json,train_results.json}` + `run_metadata_session1.json`. Việc tiếp theo: `python launch_detached.py session2` (gen2 → SPIN-2 → eval).
- 22/09/2026: session1 xong (train_loss 0.683). Spawn session2 (`fc-01M34Q53H8C2MZ9VFSFPBSTWD4`): gen2 từ `spin1` → `spin2_data` → SPIN-2 → eval 4 models. Đang chạy detached.
- 22/09/2026: **SESSION2 DONE** 15:45 UTC (còn $0.25, vừa kịp). `spin2` 238/238 loss 0.684 (~47m), `evaluation/summary.json` base 6.0% / sft 33.6% / spin1 32.8% / spin2 34.8% (n=500). Volume đủ 12 artifacts, `run_metadata_session2.json` đã commit. GPU release.
- 22/09/2026: phân tích hậu kiểm SFT vs SPIN-2 (n=500): cùng đúng 153, cùng sai 311, discordant b=15 (SFT đúng→SPIN2 sai) vs c=21 (ngược lại), net +6. McNemar χ²=0.69 (Yates) p=0.405, exact binomial p=0.405 → không significant. Bootstrap paired 20k: diff +1.20 pp, 95% CI [-1.20,+3.60] pp, P(diff>0)=81.9%.
- 23/09/2026: viết code SPIN-3/4 (copy pattern `stage_session2`, giữ `k=1`/official untouched/resume-by-ckpt) + fix review (timeout 3h, skip-eval via `_summary_has`). Tải Volume nick cũ về `artifacts_full/` (109 files, từng kẹt 99% 3 file lớn rồi tự về đích). Đẩy sang Volume nick mới đủ 12/12. `deploy` FAIL cổng payment A100 (dù Starter còn $30/$30) — user tự xử lý với Modal.
- 23/09/2026: chuyển sang workspace bạn `nhu-y-84` ($16.60, đã verify A100). `modal token new` + `profile activate nhu-y-84` → tạo `spin-artifacts` mới, `volume put` đủ 15/15 mục (9 dirs + 6 files lẻ, `evaluation`+`run_metadata_session2` trước để probe cú pháp `put <dir> /` → `/<dir>`). Verify sâu: `sft` (ckpt-250/375+model.safetensors), `spin2` (ckpt-238+model.safetensors), `data` đủ. `deploy` PASS 137s (image build OK, qua cổng A100). Spawn `session3` detached `fc-01M36PAYP1GG3JGY54RCVNXFS0` (gen `spin2`→`gen_iter2`→`spin3_data`→SPIN-3→eval 5 model). Lưu ý: hàm `status` trên cloud chỉ in `/artifacts` (nghi truncate output) — verify bằng `volume ls` local mới chuẩn.
- 23/09/2026: user thắc mắc eval thừa (script ghi đè `summary.json` + eval lại model cũ, phí ~$1.8 ở session3). Không kill (commit 1 lần cuối stage — kill mất gen+train ~$3.5). Fix cho `session4` (chưa chạy): (1) `eval_gsm8k_vi.py` merge mode + `--force` (skip model đã có accuracy, test local pass 2/2 skip, summary giữ nguyên); (2) `app.py` thêm `commit_volume()` sau gen/convert/train của `session4`; (3) `modal_app.py` thêm gói `modal` vào image. Verify `py_compile` 10/10 + YAML OK. Deploy bản mới PASS 75s (job session3 cũ chạy snapshot cũ, không ảnh hưởng). Chờ session3 DONE mới spawn session4.
- 23/09/2026 17:13+07: **SESSION3 DONE** (task 0, GPU release) nhưng kết quả INVALID: `spin3` 2.0%. Điều tra: (a) `summary.json` mới cho sft 29.6/spin1 24.8/spin2 2.2 (cũ: 33.6/32.8/34.8), base giữ nguyên 6.0% → chỉ model từ Volume là lệch; (b) `preds_spin2`≈`preds_spin3` y hệt nhau 3 dòng đầu + `gen_iter2` toàn `Answer:\n<số>` trơ trụi (có váng `</think>`) → weights spin2 trên Volume mới đã hỏng; (c) safetensors local header/size OK nhưng **spot-check CPU local `spin2` cũng ra rác** (`Answer:\n` trống) → file local đã hỏng từ lúc tải về 23/09 (đúng 3 file kẹt 99%: sft/spin1/spin2 model.safetensors 1.2GB). Kết luận: download nhiễm, upload trung thực file nhiễm. Volume nick cũ (`huynhngoccung74`) còn nguyên `spin-artifacts` — đường cứu. CHƯA spawn session4 (spin3 nhiễm độc từ spin2 hỏng). Kế hoạch: tải lại 3 safetensors từ Volume cũ (verify CPU từng cái) → thay trên Volume mới → xóa `spin3*` nhiễm → chạy lại session3 (code mới skip eval thừa) → rồi session4.
- 23/09/2026 tối: PHỤC HỒI XONG. Tải lại 3 safetensors từ Volume cũ về `TEMP\spinfix\` (spin2 nhanh, sft+spin1 chậm — user sốt ruột abort 1 lần, tải lẻ từng file thì xong). Verify CPU cả 3 bằng prompt Việt chuẩn: ra `Solution:\nReason:\n...` đàng hoàng (spin2-test còn giải đúng 54-9-28=17). Thay lên Volume mới (`put -f` từng file), xóa sạch đồ nhiễm (`spin3/ spin3_generation/ spin3_data/ spin3_config.yaml run_metadata_session3.json`), khôi phục `summary.json`+4 `preds_*.jsonl` gốc, sửa luôn backup local. Billing: metered $17.44 / credits -$17.29 → còn ~$11-12, đủ chạy lại session3 (~$4, code mới chỉ eval spin3) + session4 (~$4.5). CHỜ user gật mới spawn lại.
- 23/09/2026 tối: user chất vấn checkpoint 1/2 có chuẩn không + tiền mất oan (~2h A100 ≈ $5-5.5 cho session3 invalid — lỗi tôi: upload không verify nội dung). Chứng minh: (1) CPU full-logic 50 mẫu: spin2 mới 36.0% (khớp gốc 34.8%); (2) thêm hàm `spottest` (T4 rẻ) vào `modal_app.py`, GPU spot 50 mẫu logic eval chuẩn: **spin1 36.0% + spin2 36.0%** (file hỏng chỉ 2%). Fix kèm: docstring `status()` bị gộp dòng gây IndentationError (sửa xong, compile OK). Volume hiện sạch. Chờ lệnh spawn lại session3.
- 23/09/2026 tối: user yêu cầu chắc chắn cả `sft` → GPU spot 50 mẫu: **sft 34.0% (17/50)**, khớp gốc 33.6%. Cả 3 checkpoint (sft/spin1/spin2) đã xác minh xong. Sẵn sàng spawn lại session3 (chỉ eval spin3) rồi session4 (chỉ eval spin4).
- 23/09/2026 tối: user gật → respawn `session3` detached `fc-01M3711JW9K8SE7S6N4HV2SB1M` (Volume đã sạch + code merge/commit mới; dự ~1.5h: gen+train+chỉ eval spin3). Xong sẽ spawn `session4` (~1.5h).
- 23/09/2026 tối: **SESSION3 (chạy lại) DONE** (`run_metadata_session3.json` 13:18 UTC, GPU release, không còn task GPU). Train 238/238 steps (~40.6 phút, loss 0.6854). Kết quả VALID: `summary.json` base 6.0% / sft 33.6% / spin1 32.8% / spin2 34.8% / **spin3 34.2%** (n=500) — merge mode giữ nguyên 4 số cũ, chỉ thêm spin3. Spin3 kém spin2 nhẹ -0.6pp. Billing workspace: metered $22.24 / credits -$22.05 → billed $0.00. Việc tiếp theo: spawn `session4`.
- 23/09/2026 tối: spin2→spin3 -0.6pp là nhiễu (discordant b=23/c=20, net -3 mẫu; McNemar χ²=0.09, p≈1.0; 95% CI [-3.17,+1.97]pp). User lệnh chạy luôn → deploy mới PASS 24.6s (image cached) + spawn `session4` detached `fc-01M377A04AMK2T9YJXMWHWCSH8` (gen `spin3`→`gen_iter3`→`spin4_data`→SPIN-4→chỉ eval spin4, dự ~1.5h).
- 23/09/2026 tối: user nghi kết quả spin3 → xác minh 5 lớp, KẾT LUẬN ĐÚNG: (1) recompute `preds_spin3.jsonl` 171/500=34.2% khớp summary; (2) train đủ 238 steps/2 epochs (trainer_state); (3) `gen_iter2` 4000 dòng đúng format Solution/Reason/Answer (không phải rác `Answer:\n<số>` như lần nhiễm); (4) raw output là lập luận tiếng Việt đàng hoàng; (5) T4 spottest cùng 50 mẫu: spin2 36% (18/50) vs spin3 40% (20/50) — spin3 còn nhỉnh hơn trên subset này. Tổng: spin2≈spin3, chênh full-500 chỉ là nhiễu subset.
- 23/09/2026 tối: **SESSION4 DONE** (`run_metadata_session4.json` 15:05 UTC, GPU release). Train 238/238 loss 0.6875. Eval FINAL (n=500): base 6.0% / sft 33.6% / spin1 32.8% / spin2 34.8% / spin3 34.2% / **spin4 32.6%**. spin2→spin4 -2.2pp (net -11 mẫu, b=24/c=13, McNemar χ²=2.70 < 3.84, 95% CI [-4.58,+0.18]pp) → vẫn là nhiễu, nhưng xác nhận plateau: loss 4 vòng 0.683→0.688 đi ngang, SPIN chỉ +~1pp trên SFT rồi bão hòa ở trần 0.6B. PROJECT HOÀN TẤT.
- 23/09/2026 tối: user lệnh lưu session + đóng GPU → SESSION.md chốt tại đây; `modal app stop` deployed app (0 task, GPU đã release từ khi session4 exit). Muốn chạy tiếp (serve/demo/spottest) phải `deploy` lại.
