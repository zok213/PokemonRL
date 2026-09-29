# Kịch Bản Thuyết Trình Seminar & Bộ Câu Hỏi Bảo Vệ Hội Đồng (2026–2027)
## Chủ Đề: Autonomous Decision-Making in Long-Horizon JRPGs: From Deep RL to Foundation Models

**Đối tượng sử dụng:** Sinh viên / Học viên cao học báo cáo chuyên đề trước Giáo sư và Lớp học  
**Thời lượng dự kiến:** 20 – 25 phút thuyết trình + 10 phút Q&A phản biện  
**Học kỳ áp dụng:** Semester 2026 – 2027  
**Tài liệu tham chiếu chính:** [autonomous_rpg_agents_comprehensive_survey_2026.md](file:///C:/Users/Admin/.gemini/antigravity/brain/8cc20e3b-7f6e-468d-87b5-eaa2aae2c87a/autonomous_rpg_agents_comprehensive_survey_2026.md)

---

## PHẦN 1: CẤU TRÚC 12 SLIDE BÁO CÁO CHI TIẾT (SLIDE-BY-SLIDE GUIDE)

### Slide 1: Trang Tiêu Đề & Giới Thiệu
* **Tiêu đề:** Quyết Định Tự율 Trong Môi Trường Nhập Vai Dài Hạn (Long-Horizon JRPGs): Từ Deep Reinforcement Learning Đến Foundation Models.
* **Người thực hiện:** [Họ và tên sinh viên / Nhóm nghiên cứu]
* **Giáo viên hướng dẫn:** [Tên Thầy/Cô]
* **Speaker Notes (Lời thoại gợi ý):**
  > *"Kính thưa Thầy và các bạn, hôm nay nhóm em xin trình bày bài tổng quan nghiên cứu (Literature Review) về một trong những bài toán hóc búa nhất của Trí tuệ Nhân tạo hiện đại: Giải quyết quá trình ra quyết định tuần tự trong không gian trạng thái cực lớn, phần thưởng siêu thưa và chân trời thời gian lên tới hàng trăm ngàn bước qua testbed kinh điển: Pokémon Red."*

---

### Slide 2: Giới Hạn Của Atari & Định Lý Sụp Đổ Chân Trời (The Horizon Collapse)
* **Nội dung hiển thị:**
  * Bảng so sánh MuJoCo ($T \sim 1{,}000$), Atari 2600 ($T \sim 3{,}000$), Montezuma's Revenge ($T \sim 5{,}000$) vs. Pokémon Red ($T > 300{,}000$).
  * Công thức $\tau_{\text{eff}} \approx \frac{1}{1-\gamma}$.
  * Định lý suy hao gradient: $\|\nabla_\theta \mathcal{L}_{PPO}\| \le C \cdot (\gamma \lambda)^K \cdot |R^*| \to 0$ khi $K = 25{,}000$.
* **Speaker Notes:**
  > *"Trong suốt một thập kỷ qua, cộng đồng RL chủ yếu tự hào với các chiến thắng trên Atari hay MuJoCo. Nhưng thực tế, Atari là bài toán 'đồ chơi' với phần thưởng dày đặc và chỉ kéo dài vài ngàn khung hình. Trong Pokémon, không gian trạng thái lên tới $2^{131072}$ cấu hình RAM, và để lấy được một huy hiệu phải mất hàng chục ngàn bước di chuyển và hội thoại. Về mặt toán học, hệ số chiết khấu $\gamma$ khiến tín hiệu gradient từ phần thưởng suy giảm về 0 trước khi kịp lan truyền về các bước đi ban đầu. Flat Model-Free RL chắc chắn thất bại nếu không có phân rã cấu trúc."*

---

### Slide 3: Trường Phái 1: Model-Free DRL & Hard Exploration (Whidden V1/V2, Pleines et al. IEEE CoG 2025)
* **Nội dung hiển thị:**
  * Sơ đồ tò mò thị giác: Pixel KNN $\to$ Hiện tượng bị thôi miên bởi sóng nước (Noisy TV Water Hypnosis).
  * Giải pháp RAM Coordinate Hashing: Đếm tọa độ $(x, y, \text{map\_id})$ qua WRAM `0xD362`, `0xD361`, `0xD35E`.
  * **Tử huyệt:** The Healing Trap (Nurse Joy loop) và bức tường Cerulean City.
* **Speaker Notes:**
  > *"Bài báo IEEE CoG 2025 của Pleines et al. đã giải quyết được lỗi tò mò thị giác của Peter Whidden bằng cách băm trực tiếp tọa độ RAM. Tuy nhiên, bài báo này gặp hai vấn đề chí mạng: Thứ nhất là 'Bẫy hồi máu' – do phần thưởng hồi máu lớn hơn nhiều phần thưởng khám phá, mạng học cách liên tục tự làm mình bị thương rồi quay lại gặp Y tá Joy. Thứ hai, mạng bị chặn cứng ở Cerulean City (chỉ đạt ~20% game) vì không thể vượt qua chuỗi nhiệm vụ phi tuyến tính."*

---

### Slide 4: Trường Phái 2: Hierarchical RL & Action Engineering (PokeRL 2026, PufferLib)
* **Nội dung hiển thị:**
  * PokeRL (Mudireddy & Patibandla, arXiv:2604.10812): Dynamic Action Masking (Khóa phím điều hướng khi đọc text `0xCF13`, khóa phím START khi đi đường).
  * Hình phạt Spam Entropy: Tăng tỷ lệ di chuyển hợp lệ từ 27.2% lên 68.2%.
  * Rubinstein PufferLib: 25+ hàm thế năng RAM & sự thật về code hack Safari Zone.
* **Speaker Notes:**
  > *"Để ngăn chặn agent bấm loạn xạ hoặc mở/đóng menu ở tần số 30 Hz để né phạt, PokeRL 2026 đã đưa vào cơ chế lọc hành động theo ngữ cảnh RAM. Đồng thời, Rubinstein sử dụng PufferLib để vector hóa C đạt 100k bước/giây và hoàn thành cả 8 Gym. Nhưng chúng ta cần nhìn nhận trung thực: PufferLib đã phải 'gian lận' bằng script Python can thiệp đóng băng bộ đếm 500 bước trong Safari Zone. Đây không phải là AGI tự học thực sự."*

---

### Slide 5: Trường Phái 3: Multi-Agent LLMs & In-Context RL (PokéAI, PokéLLMon)
* **Nội dung hiển thị:**
  * PokéAI (arXiv:2506.23689): Kiến trúc Planner-Actor-Critic tích hợp Vector DB chứa Walkthrough. Tỉ lệ thắng trận 80.8%.
  * PokéLLMon (arXiv:2402.01118): In-Context RL (ICRL) + Knowledge-Augmented Generation (KAG) trên Pokemon Showdown.
  * Hội chứng 'Panic Cascade' (Suy sụp nhận thức khi bị đối thủ đánh đòn bất ngờ).
  * Rào cản chi phí và độ trễ: 5–15 giây/bước, tốn hàng ngàn USD tiền API.
* **Speaker Notes:**
  > *"Khi chuyển sang trường phái LLM Agent, khả năng hiểu ngôn ngữ và bảng thuộc tính tương khắc của mô hình là vượt trội. PokéAI đạt tỉ lệ thắng 80.8% trong các trận đấu dã ngoại. Tuy nhiên, LLM có hai nhược điểm chết người: Thời gian phản hồi lên tới vài giây mỗi phím bấm, chi phí API lên tới hàng ngàn USD cho một lượt chơi, và đặc biệt là hiện tượng 'Panic Cascade' – khi bị đối thủ đánh một đòn chí mạng bất ngờ, LLM bắt đầu hoảng loạn, sinh ra ảo giác và liên tục đổi Pokémon qua lại cho đến khi thua cuộc."*

---

### Slide 6: Trường Phái 4: Offline Sequence Transformers (Metamon RLC 2025)
* **Nội dung hiển thị:**
  * Replay Inversion: Chuyển đổi 22 triệu trận đấu Showdown công khai thành góc nhìn POMDP thứ nhất.
  * AMAGO Transformer: Chú ý nội bộ lượt đấu (Intra-turn) + Transformer nhân quả liên lượt đấu (Inter-turn).
  * Kết quả: Đạt Top 10% Elo người chơi thật, suy luận $<15$ ms trên GPU phổ thông, 0 USD phí API.
* **Speaker Notes:**
  > *"Để khắc phục sự chậm chạp và tốn kém của LLM, bài báo Metamon tại RLC 2025 đã biến bài toán chiến đấu thành Sequence Modeling qua 22 triệu replay của người chơi thật. Nhờ deterministic inversion, Metamon đạt tốc độ phản hồi dưới 15 mili-giây, vượt qua mốc Elo 1500 của con người mà không tốn một đồng chi phí token nào. Đây chính là mảnh ghép hoàn hảo cho module chiến thuật."*

---

### Slide 7: Trường Phái 5: High-Performance Systems & Compiler Auto-Gen (PokeAgent, PokeJAX COLM 2026)
* **Nội dung hiển thị:**
  * Nút thắt mô phỏng: Python multiprocessing (1k SPS) vs. PokeJAX GPU (15.2M SPS).
  * Bộ dịch tự động Agentic Compiler: Chuyển đổi code Python/TypeScript sang JAX/Rust với chi phí $<10$ USD.
  * Quy trình kiểm định 4 tầng: Property Tests $\to$ Interaction Tests $\to$ Rollout Divergence $\to$ Sim-to-Sim Transfer.
* **Speaker Notes:**
  > *"Ở trường phái hệ thống, công trình tại NeurIPS 2025 và COLM 2026 của nhóm Chi Jin đã chứng minh: Nút thắt lớn nhất của RL hiện đại nằm ở IPC overhead. Bằng cách dùng LLM compiler tự động dịch logic Game Boy sang mảng JAX chạy thẳng trong VRAM GPU, tốc độ mô phỏng đạt trên 15 triệu bước/giây, đưa tỷ lệ lãng phí mô phỏng xuống dưới 4%."*

---

### Slide 8: Giải Phẫu 4 Bệnh Lý Thuật Toán Kinh Điển (Pathologies Post-Mortem)
* **Nội dung hiển thị:**
  1. *The Healing Trap:* Tối ưu hóa chu trình ngắn vì $r_{\text{heal}} \gg r_{\text{nav}}$.
  2. *The Noisy TV:* Entropy sóng nước đánh lừa hàm tò mò pixel.
  3. *Menu Locks:* Mở START menu để đóng băng bộ đếm thời gian phạt.
  4. *The 500-Step Safari Zone:* Giới hạn bước chân khiến xác suất mò ngẫu nhiên $P < 10^{-14}$.
* **Speaker Notes:**
  > *"Nhìn lại toàn bộ lịch sử, mọi agent thất bại đều do vấp phải 4 bệnh lý này. Hiểu rõ nguyên nhân toán học của 4 căn bệnh này là chìa khóa để chúng ta không lặp lại sai lầm khi thiết kế hệ thống mới."*

---

### Slide 9: Đề Xuất Kiến Trúc Hợp Nhất SOTA 2026–2027 (The Unified Blueprint)
* **Nội dung hiển thị:**
  * Sơ đồ 3 tầng:
    * Tầng 1: Go-Explore State Archiving (Lưu snapshot 32 KB, return-then-explore để giải Safari Zone không cần hack).
    * Cầu nối: Dynamic Action Masking (Khóa menu, khóa text).
    * Tầng 2: Overworld Navigator (PokeJAX + Critic-Free GRPO) + Decoupled Metamon Combat Engine.
* **Speaker Notes:**
  > *"Từ những phân tích trên, nhóm em đề xuất kiến trúc hợp nhất cho chu kỳ 2026–2027: Dùng Go-Explore lưu checkpoint trạng thái để giải Safari Zone hoàn toàn trung thực; dùng GRPO trên các ngã rẽ để loại bỏ hoàn toàn mạng Critic và hiện tượng trôi giá trị; và tách riêng module chiến đấu cho Metamon xử lý ở tốc độ 15 mili-giây."*

---

### Slide 10: Bảng So Sánh Đối Đầu Toàn Diện (Master Comparative Benchmark)
* **Nội dung hiển thị:**
  * Bảng ma trận đối đầu giữa cả 5 trường phái dựa trên: Throughput (SPS), Inference Latency, API Cost, Deepest Progress, Hardware Requirement.
* **Speaker Notes:**
  > *"Bảng đối chiếu này cho thấy không có một mô hình đơn lẻ nào là hoàn hảo: DRL mạnh về tốc độ nhưng yếu về suy luận dài hạn; LLM mạnh về suy luận nhưng quá chậm và đắt; Transformer offline mạnh về chiến đấu; và JAX mạnh về hạ tầng. Tương lai của AGI trong game thế giới mở bắt buộc phải là Neuro-Symbolic Hybrid."*

---

### Slide 11: Lộ Trình Mở Rộng & Khả Năng Tổng Quát Hóa (Future Roadmap)
* **Nội dung hiển thị:**
  * Khả năng chuyển giao (Cross-Game Transfer) sang Zelda, Final Fantasy, Dragon Quest.
  * Sim-to-Real: Ứng dụng quản lý tài nguyên và giải quyết bài toán chuỗi cung ứng dài hạn.
  * An toàn AI: Kiểm soát hành vi lặp vô tận và bẫy cục bộ.
* **Speaker Notes:**
  > *"Các giải pháp chúng ta phát triển trên Pokémon Red không chỉ để chơi game. Cấu trúc quest dài hạn, tài nguyên hữu hạn và quyết định không thể đảo ngược chính là mô hình thu nhỏ của các bài toán điều phối robot tự hành và tối ưu hóa chuỗi logistics trong đời thực."*

---

### Slide 12: Kết Luận & Phiên Hỏi Đáp (Conclusion & Q&A)
* **Nội dung hiển thị:**
  * Tóm tắt 3 thông điệp chính: (1) Flat RL đã hết thời; (2) Cần kết hợp Go-Explore + GRPO + Offline Transformer; (3) Hệ thống mã nguồn đã được kiểm chứng.
  * Link repo mã nguồn và bản khảo cứu chi tiết.
  * Lời cảm ơn.
* **Speaker Notes:**
  > *"Em xin chân thành cảm ơn Thầy và các bạn đã chú ý lắng nghe. Nhóm em rất mong nhận được những câu hỏi và ý kiến đóng góp từ Thầy và cả lớp!"*

---

## PHẦN 2: BỘ CÂU HỎI VÀ ĐÁP ÁN BẢO VỆ PHẢN BIỆN (PROFESSOR Q&A DEFENSE SCRIPT)

Dưới đây là 5 câu hỏi sắc bén nhất mà Giáo sư hoặc người nghe có chuyên môn cao về RL/AI thường hỏi, cùng câu trả lời chuẩn mực của một AI Research Scientist:

---

### Câu hỏi 1 của Thầy: *"Tại sao em không đơn giản là tăng hệ số chiết khấu $\gamma$ lên $0.9999$ hoặc $0.99999$ để giải quyết bài toán chân trời dài hạn (Long-Horizon) thay vì phải làm phức tạp hóa kiến trúc?"*
* **Cách trả lời chuẩn chuyên gia:**
  > *"Dạ thưa Thầy, về mặt lý thuyết thuần túy, khi $\gamma \to 1$, chân trời hiệu dụng $\tau_{\text{eff}} \approx \frac{1}{1-\gamma}$ sẽ tiến tới vô cùng. Tuy nhiên, trong thực nghiệm Deep RL, việc tăng $\gamma$ lên quá sát $1.0$ gây ra ba hậu quả toán học nghiêm trọng:*
  > 1. ***Phương sai của ước lượng Bellman bùng nổ (Variance Explosion):** Mục tiêu học của mạng Critic $y_t = r_t + \gamma V(s_{t+1})$ tích lũy sai số xấp xỉ qua hàng ngàn bước, khiến hàm mất mát của Critic bị phân kỳ (Divergence).*
  > 2. ***Tốc độ co của toán tử Bellman bị triệt tiêu:** Hệ số co của ánh xạ Bellman là $\gamma$. Khi $\gamma = 0.9999$, tốc độ hội tụ của thuật toán lặp giá trị giảm đi hàng trăm lần.*
  > 3. ***Hiện tượng trôi giá trị (Value Drift):** Mạng nơ-ron sâu không thể phân biệt được sự khác biệt giữa trạng thái có Return $1500.2$ và $1500.5$ qua một mạng tích chập thông thường. Do đó, cộng đồng nghiên cứu quốc tế từ Sutton, Precup đến DeepMind đều khẳng định: Không thể giải quyết Long-Horizon bằng cách tăng $\gamma$, mà bắt buộc phải phân rã cấu trúc thời gian (Temporal Abstraction) hoặc dùng mô hình phân tầng (Hierarchical Options).*"*

---

### Câu hỏi 2 của Thầy: *"Repo của David Rubinstein (drubinstein/pokemonred_puffer) đã hoàn thành cả game và đánh bại Tứ Đại Thiên Vương. Vậy nghiên cứu thêm về Pokémon Red có còn giá trị học thuật không, hay chỉ là làm lại những gì người ta đã làm?"*
* **Cách trả lời chuẩn chuyên gia:**
  > *"Dạ thưa Thầy, đây là một điểm cực kỳ tinh tế mà nhóm em đã giải phẫu rất kỹ trong phần 'Pathologies Post-Mortem':*
  > * Rubinstein đã có đóng góp rất lớn về mặt **Kỹ thuật hệ thống (Systems Engineering)** khi tối ưu PufferLib đạt 100k bước/giây.*
  > * **Tuy nhiên, về mặt Trí tuệ Nhân tạo Tổng quát (AGI), công trình đó KHÔNG giải quyết bài toán bằng RL thuần túy.** Tác giả đã phải nhồi vào hơn 25 hàm thế năng phần thưởng thủ công (Handcrafted Potentials) đọc trực tiếp từ WRAM, nghĩa là con người đã 'mớm' sẵn đường đi cho mô hình.*
  > * **Đặc biệt, tại Safari Zone, tác giả đã dùng script can thiệp vào RAM để đóng băng bộ đếm 500 bước chân và tự động xịt Repel.** Nếu tắt đoạn script này đi, mô hình PufferLib lập tức bị kẹt vĩnh viễn và không bao giờ phá đảo được.*
  > * Do đó, giá trị học thuật hiện nay nằm ở chỗ: **Làm sao để một hệ thống tự giải được toàn bộ game một cách trung thực (Autonomous & Zero-Cheat)** thông qua Go-Explore Checkpointing và suy luận bán biểu trưng, chứ không phải dùng mẹo can thiệp bộ nhớ.*

---

### Câu hỏi 3 của Thầy: *"Tại sao không đưa thẳng toàn bộ khung hình hoặc prompt vào các mô hình Frontier LLM như GPT-4o, Claude 3.5 Sonnet hay Gemini 1.5 Pro để chúng chơi từ đầu đến cuối?"*
* **Cách trả lời chuẩn chuyên gia:**
  > *"Dạ thưa Thầy, phương án dùng thuần LLM gặp phải 3 rào cản bất khả thi trong thực tế:*
  > 1. ***Độ trễ suy luận (Latency):** Mỗi frame của Game Boy chạy ở 60 Hz, khi frame-skip là 16 thì agent cần ra quyết định mỗi 250 mili-giây. Một lệnh gọi API LLM mất từ 3 đến 10 giây. Để chơi xong 300,000 bước, một con bot LLM sẽ mất hơn 35 ngày chạy liên tục không ngừng nghỉ.*
  > 2. ***Chi phí tài chính (Financial Cost):** Mỗi bước gửi kèm ngữ cảnh tiêu tốn trung bình 1,000 tokens. Với 300,000 bước, chi phí API cho duy nhất MỘT lượt chơi thử là hơn 3,000 USD. Nếu huấn luyện hoặc lặp lại thực nghiệm, chi phí là không tưởng.*
  > 3. ***Hội chứng Panic Cascade (Suy sụp nhận thức):** Trong nghiên cứu The PokeAgent Challenge (NeurIPS 2025), khi gặp tình huống bất ngờ (bị đánh critical hit), prompt context của LLM bị mất cân bằng, dẫn đến việc LLM liên tục ảo giác sinh ra các phím bấm không hợp lệ hoặc rơi vào vòng lặp đổi Pokémon qua lại cho đến chết.*
  > * Vì vậy, giải pháp tối ưu là chỉ dùng LLM ở tầng Macro-Planner (vài ngàn bước mới gọi một lần để định hướng nhiệm vụ), còn tầng điều khiển động cơ vi mô phải giao cho DRL và Offline Transformer.*"

---

### Câu hỏi 4 của Thầy: *"Ý tưởng kết hợp Go-Explore Checkpointing và Critic-Free GRPO của nhóm có gì mới so với các thuật toán phân tầng truyền thống như FeUdal Networks hay Option-Critic?"*
* **Cách trả lời chuẩn chuyên gia:**
  > *"Dạ thưa Thầy, các kiến trúc phân tầng cổ điển như FeUdal Networks hay Option-Critic đều dựa trên việc học đồng thời cả Policy phân tầng lẫn Value Critic ở từng tầng. Điều này dẫn đến sự mất ổn định cực lớn vì mục tiêu của tầng trên thay đổi liên tục khiến tầng dưới không thể hội tụ (Non-stationarity).*
  > * Đột phá của đề xuất nhóm em nằm ở hai điểm:*
  > 1. ***Go-Explore Checkpointing giải quyết triệt để vấn đề Detachment:** Thay vì hy vọng mạng tự tìm lại được lối vào Safari Zone hay Cinnabar Island, hệ thống lưu trữ trực tiếp trạng thái RAM 32 KB của Game Boy vào Archive. Khi gặp thất bại, agent được 'dịch chuyển' trực tiếp về biên giới khám phá sâu nhất mà không gặp rủi ro chệch hướng (Derailment).*
  > 2. ***GRPO loại bỏ hoàn toàn mạng Critic:** Bằng cách nhân bản trạng thái tại ngã rẽ thành $G=8$ nhánh song song và tính ưu thế tương đối giữa các nhánh (Group Relative Advantage), chúng ta không cần khởi tạo mạng Critic $V(s)$. Điều này triệt tiêu hoàn toàn hiện tượng Value Divergence trong game dài hạn và tiết kiệm 40% bộ nhớ GPU.*"

---

### Câu hỏi 5 của Thầy: *"Để đăng ký đề tài cho môn học này, nhóm em dự định chọn bài báo nào làm tâm điểm, và kế hoạch thực nghiệm ra sao?"*
* **Cách trả lời chuẩn chuyên gia:**
  > *"Dạ thưa Thầy, nhóm em xin đề xuất 2 phương án tùy thuộc vào định hướng đánh giá của Thầy:*
  > * **Phương án 1 (Tập trung chuyên sâu - Deep-Dive Benchmark Paper):** Đăng ký chính thức bài báo **Pleines et al. (IEEE Conference on Games 2025 - arXiv:2502.19920)** kết hợp mã nguồn **PokeRL (arXiv:2604.10812)**. Mục tiêu phản biện là chứng minh tại sao Pleines bị kẹt ở Cerulean City và thực nghiệm bổ sung Dynamic Action Masking để vượt qua mốc này.*
  > * **Phương án 2 (Đề tài Tổng quan Hệ thống - SOTA System Survey & Benchmark):** Đăng ký đề tài dạng **Comprehensive Literature Review & Unified Neuro-Symbolic Testbed** bao quát cả 5 trường phái từ IEEE, ACM đến NeurIPS 2025/2026. Nhóm em đã chuẩn bị sẵn khung mã nguồn Python tích hợp cả Action Masking, Go-Explore Checkpoint và GRPO Rollout Engine. Nhóm em sẵn sàng nộp bản thảo bài báo hoàn chỉnh theo format IEEEtran (đã được soạn thảo chi tiết).*
  > * Nhóm em xin lắng nghe ý kiến chỉ đạo của Thầy để chọn phương án phù hợp nhất với tiêu chí của học phần ạ!*"*
