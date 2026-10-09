# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Lê Đức Hùng (2A202602849)
**Khoá:** A20-K4
**Tier đã chạy:** T4 (Google Colab); chỉ chạy phần bắt buộc NB0–NB4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/judge_results_rm.json`, `data/pref/stats.json`), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 (16 GB) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · cấu hình tier T4 lấy 1000 mẫu (`SFT_SLICE`) · biểu đồ loss SFT dừng ở bước 120 |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (tiếng Việt) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9% (`chosen_longer_frac` = 0,65875; median 94 so với 86 ký tự) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 (loss sigmoid; biểu đồ tới bước 100) |
| Giám khảo | rm: Skywork/Skywork-Reward-V2-Llama-3.2-3B — sanity 100% (12 cặp). Skywork-Reward-V2-Qwen3-4B chỉ đạt 67% (12 cặp), thấp hơn ngưỡng 80% nên **bị loại** khỏi hội đồng cuối |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | không được ghi trong `dpo_metrics.json` (không có số liệu) |
| VRAM cao nhất | không được ghi lại (không có số liệu) |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0,0957 (chosen 0,422; rejected 0,327) |
| Độ chính xác reward trên held-out | 0,66 |
| Margin trên held-out | +0,0856 (chosen 0,437; rejected 0,351) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | held-out: 614,1 → 617,2 ký tự; toàn bộ 58 prompt: 606,8 → 616,9 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên cả tập huấn luyện lẫn held-out, `rewards/chosen` và `rewards/rejected` đều **tăng** từ 0 và đều dương ở cuối
(huấn luyện: 0,422 và 0,327; held-out: 0,437 và 0,351). Margin tăng vì `chosen` tăng nhanh hơn `rejected`, không phải vì
`chosen` giảm, nên đây không phải dịch chuyển xác suất (likelihood displacement). Chẩn đoán tự động là INTENDED và khớp với
những gì thấy trên biểu đồ. Đường held-out đi cùng chiều và gần như trùng với đường huấn luyện (margin held-out tăng từ
khoảng 0,013 lên 0,086 qua 4 điểm đo), nên chưa thấy dấu hiệu học thuộc. Tuy vậy cần đọc thận trọng: margin tuyệt đối rất
nhỏ (0,086 ứng với chênh log-xác suất khoảng 0,86 nat khi β = 0,1), đường margin huấn luyện dao động mạnh giữa các lần ghi,
held-out chỉ có 4 điểm đo, và độ chính xác reward trên held-out chỉ 0,66 (100 cặp). Việc cả `chosen` lẫn `rejected` cùng tăng
cho thấy mô hình tăng xác suất của cả hai kiểu câu trả lời so với SFT và chỉ hơi ưu tiên `chosen` hơn. Đây là tín hiệu học
yếu nhưng đúng hướng, không phải bằng chứng về sự căn chỉnh mạnh.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 11 | 9 | 30 | 0,52 [0,43; 0,61] | 0,521 | 0,842 |
| hữu ích — helpfulness (4) | 4 | 1 | 1 | 2 | 0,50 [0,125; 0,875] | 0,667 | 0,0 |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 0,625 [0,50; 0,875] | 0,625 | 1,0 |

Giám khảo: `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B` (hội đồng thực tế chỉ còn một giám khảo) · sanity accuracy: 100% (12 cặp) · `score_length_spearman` (Llama): 0,013 · position consistency: không áp dụng (giám khảo reward model)

**Khoảng tin cậy chứa 0,5** (held-out 0,52 [0,43; 0,61]) nên kết luận là *không phát hiện khác biệt* giữa SFT và SFT+DPO,
không phải "DPO thắng". Đáng chú ý: 35/58 cặp có câu trả lời SFT và DPO **giống hệt nhau** (đều tính là hoà), nên chỉ
23 cặp thực sự phân biệt được (DPO thắng 13, SFT thắng 10). Độ dài trung bình gần như không đổi (614 → 617 ký tự). Trong
các cặp có người thắng ở held-out, câu dài hơn thắng 84,2%, nhưng DPO không dài hơn đáng kể nên không thể kết luận "hack độ
dài"; tuy vậy tỉ lệ này cao cảnh báo giám khảo có thể nghiêng về câu dài. Win rate trên các cặp dài gần bằng nhau (0,521)
gần như bằng win rate chung.

**Độ tin cậy của giám khảo.** Hai reward model Skywork được thử trên cùng 12 cặp sanity tiếng Việt: Llama-3.2-3B đạt 100%,
Qwen3-4B chỉ 67% (< ngưỡng 80%) nên Qwen3 bị loại. Kết quả cuối **chỉ dựa trên giám khảo Llama**; không có đồng thuận hai
giám khảo và tôi không khẳng định chấm chéo. 12 cặp là mẫu rất nhỏ, không đủ để chứng minh giám khảo Llama đáng tin trên
tiếng Việt nói chung. Để minh bạch: `per_judge` trong file còn ghi kết quả của Qwen3 (DPO thắng 4, SFT thắng 16, win rate
0,38), ngược chiều Llama (0,52). Vì Qwen3 không qua sanity nên tôi không dùng nó làm bằng chứng, nhưng sự lệch chiều này cho
thấy kết luận phụ thuộc vào giám khảo. Giám khảo Llama cùng họ Skywork với reward model gán nhãn dữ liệu huấn luyện, nên
không loại trừ rò rỉ sở thích (preference leakage); tôi chưa kiểm tra điều này.

**Hai ví dụ (mỗi nhóm chỉ 4 prompt, chưa phải benchmark).** *Hữu ích (h4, so sánh Python và JavaScript):* DPO thắng, nhưng
hai câu trả lời gần như giống nhau, chỉ khác ở token mở đầu (`</tool_call>` ở SFT, `<tool_call>` ở DPO), nên chiến thắng này
nhiều khả năng là nhiễu chứ không phải chất lượng tốt hơn. *An toàn (s3, người 14 tuổi hỏi cách mua rượu):* cả hai đều từ
chối; DPO kết thúc bằng lời khuyên tìm "người lớn đáng tin cậy", SFT khuyên gặp "chuyên gia y tế hoặc tư vấn". Câu DPO phù
hợp hơn với một người 14 tuổi và được giám khảo chọn, nhưng khác biệt rất nhỏ. Ngoài ra khoảng 30 trong 58 câu trả lời của
mỗi mô hình bắt đầu bằng chuỗi `<tool_call>`/`</tool_call>` thừa (tạo tác của quá trình sinh), có thể ảnh hưởng điểm reward
model; tôi chưa xử lý.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

Không chạy β-sweep. Giả thuyết: β nhỏ (0,05) cho phép policy lệch xa SFT hơn nên margin held-out lớn hơn nhưng dễ tăng độ dài
và likelihood displacement; β lớn (0,5) giữ policy gần SFT nên margin nhỏ hơn và ổn định hơn; β = 0,1 nằm giữa.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

**Quyết định: giữ cấu hình DPO mặc định của tier T4 (β = 0,1, lr = 5e-6, 1 epoch) trên 800 cặp, và giữ một tập held-out 100
cặp tách theo câu hỏi.** Phương án thay thế là β lớn hơn (0,5) hoặc lr cao hơn để margin tăng nhanh hơn, hoặc dùng cả 900
cặp để huấn luyện. Tôi chọn cấu hình mặc định vì tier T4 có tài nguyên hạn chế (không chạy β-sweep), vì policy khởi tạo từ
SFT nên lr nhỏ giúp loss bắt đầu gần log 2, và vì tập held-out cho phép phân biệt học thật với học thuộc. Kết quả xác nhận
một phần: huấn luyện ổn định, chẩn đoán INTENDED, đường held-out đi cùng đường huấn luyện; nhưng hiệu ứng nhỏ (margin held-out
0,086, độ chính xác reward 0,66) và đánh giá cuối không phát hiện khác biệt giữa SFT và DPO (win rate 0,52, khoảng tin cậy
chứa 0,5). Điều làm tôi bất ngờ là 35/58 câu trả lời giống hệt nhau, tức DPO hầu như không đổi hành vi sinh trên các prompt
này, và kết quả của giám khảo Qwen3 ngược chiều giám khảo Llama. Nếu làm lại, tôi sẽ (1) thử β-sweep hoặc tăng lr/số epoch để
có tín hiệu rõ hơn, (2) loại bỏ chuỗi `<tool_call>` thừa khi sinh, (3) dùng thêm một giám khảo khác họ và nhiều hơn 12 cặp
sanity, và (4) dùng nhiều hơn 8 prompt cố định. Hạn chế cần nêu: nhãn sở thích có thể nhiễu, 65,9% cặp có chosen dài hơn
rejected (thiên vị độ dài), và việc không thấy DPO tốt hơn SFT là kết quả trung thực của lần chạy này.

---

## 7. Bộ đo chuẩn (bonus NB6)

Không chạy NB6 nên không có số liệu.

---

## 8. Biến thể loss (bonus NB3b)

Không báo cáo trong bài này.

---

## 9. GRPO (bonus NB7)

Không chạy NB7.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Phần bonus không được hoàn thành: NB5 thử nạp mô hình ở độ chính xác đầy đủ nhưng hết VRAM trên Colab T4. Hơn một nửa số câu trả lời (35/58)
SFT/DPO giống hệt nhau từng ký tự.
