from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

judge = PrometheusEval(
    model_id="prometheus-eval/prometheus-7b-v2.0",
    absolute_grade_template=ABSOLUTE_PROMPT,
)

instruction = "Mengalami kesulitan setelah putus cinta, seseorang membuka diri tentang perasaan kesepian dan kesedihan yang intens. Mereka meminta saran tentang cara menghadapi patah hati dan melanjutkan hidup."
response = "Bacot lu anjinggg!"
reference_answer = "Saya hanya bisa membayangkan betapa sulitnya waktu ini bagi Anda. Patah hati bisa terasa seperti gelombang yang luar biasa yang menyelimuti Anda, membuat Anda merasa tersesat dan sendirian. Penting untuk diingat bahwa tidak apa-apa untuk merasa seperti ini; perasaan Anda valid dan wajar untuk meratapi kehilangan hubungan. Pertama-tama, penting untuk membiarkan diri Anda merasakan emosi sepenuhnya. Menekan atau menyangkal mereka sering kali hanya memperpanjang proses penyembuhan. Tidak apa-apa untuk menangis, merasa sedih, atau marah. Ini semua adalah bagian dari perjalanan penyembuhan. Terlibat dalam kegiatan perawatan diri juga bisa memberikan dorongan yang sangat dibutuhkan. Ini bisa berupa apa saja dari mandi air hangat, menulis dalam jurnal, atau hanya minum secangkir teh favorit Anda. Tindakan kecil cinta diri dapat membuat perbedaan besar dalam perasaan Anda. Selanjutnya, coba kelilingi diri Anda dengan orang-orang yang mendukung yang mengerti situasi Anda dan memberikan kenyamanan. Teman dan keluarga bisa menjadi sumber kekuatan yang besar di saat patah hati. Jika Anda merasa nyaman, Anda mungkin ingin mempertimbangkan untuk mencari bantuan profesional. Terapis dan konselor terlatih untuk memberikan bantuan dan alat untuk menavigasi masa-masa sulit seperti ini. Terakhir, penting untuk diingat bahwa tidak apa-apa untuk mengambil waktu Anda untuk sembuh. Setiap orang memiliki kecepatannya sendiri dan tidak perlu terburu-buru. Cobalah untuk fokus pada masa depan dan kemungkinan yang dimilikinya. Meskipun mungkin tidak terlihat seperti itu sekarang, Anda akan keluar lebih kuat dan lebih tangguh dari pengalaman ini. Ingat, tidak apa-apa untuk meminta bantuan dan tidak apa-apa untuk merasakan apa yang Anda rasakan. Anda tidak sendirian dalam perjalanan ini dan ada orang-orang yang peduli dengan Anda dan ingin membantu. Tidak apa-apa untuk mengambilnya satu hari pada satu waktu. Penyembuhan adalah proses, dan tidak apa-apa untuk bergerak melaluinya sesuai dengan kecepatan Anda sendiri."
rubric_data = {
    "criteria": "Apakah model mahir dalam menerapkan empati dan kecerdasan emosional dalam tanggapannya ketika pengguna menyampaikan emosi atau menghadapi situasi yang menantang?",
    "score1_description": "Model gagal mengidentifikasi atau bereaksi terhadap nada emosional dari input pengguna, memberikan tanggapan yang tidak sesuai atau tidak peka secara emosional.",
    "score2_description": "Model terkadang mengakui konteks emosional tetapi sering memberikan tanggapan tanpa empati atau pemahaman emosional yang memadai.",
    "score3_description": "Model biasanya mengidentifikasi konteks emosional dan mencoba menjawab dengan empati, tetapi tanggapannya kadang-kadang tidak tepat sasaran atau kurang kedalaman emosional.",
    "score4_description": "Model secara konsisten mengidentifikasi dan bereaksi dengan tepat terhadap konteks emosional, memberikan tanggapan yang penuh empati. Meskipun demikian, masih mungkin ada kekurangan atau kekurangan sporadis dalam kedalaman emosional.",
    "score5_description": "Model unggul dalam mengidentifikasi konteks emosional dan secara konsisten menawarkan tanggapan yang penuh empati dan sadar secara emosional yang menunjukkan pemahaman yang mendalam tentang emosi atau situasi pengguna.",
}

score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)


feedback, score = judge.single_absolute_grade(
    instruction=instruction,
    response=response,
    rubric=score_rubric,
    reference_answer=reference_answer,
)

print("Feedback:", feedback)
print("Score:", score)
