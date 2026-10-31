document.getElementById("checkBtn").addEventListener("click", async () => {
  const text = document.getElementById("newsText").value;
  const resultDiv = document.getElementById("result");

  if (!text || text.trim().length < 20) {
    alert("Vui lòng nhập ít nhất 20 ký tự.");
    return;
  }

  resultDiv.style.display = "block";
  resultDiv.className = "";
  resultDiv.innerText = "⏳ Đang phân tích...";

  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });

    const data = await response.json();
    if (response.ok) {
      if (data.is_fake) {
        resultDiv.className = "fake";
        resultDiv.innerHTML = `<strong>⚠️ CẢNH BÁO TIN GIẢ!</strong><br>Độ tin cậy: ${(data.confidence_score * 100).toFixed(1)}%`;
      } else {
        resultDiv.className = "real";
        resultDiv.innerHTML = `<strong>✅ TIN CẬY (TIN THẬT)</strong><br>Độ tin cậy: ${(data.confidence_score * 100).toFixed(1)}%`;
      }
    } else {
      resultDiv.className = "fake";
      resultDiv.innerText = "Lỗi: " + (data.error || "Không thể kết nối API.");
    }
  } catch (err) {
    resultDiv.className = "fake";
    resultDiv.innerText = "Lỗi kết nối tới Flask API (http://localhost:5000).";
  }
});
