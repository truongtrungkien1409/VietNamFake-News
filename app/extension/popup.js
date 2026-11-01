document.addEventListener("DOMContentLoaded", async () => {
  const DEFAULT_API_URL = "http://127.0.0.1:5000";
  const apiUrlInput = document.getElementById("apiUrlInput");
  const saveConfigBtn = document.getElementById("saveConfigBtn");
  const newsText = document.getElementById("newsText");
  const getSelectionBtn = document.getElementById("getSelectionBtn");
  const checkBtn = document.getElementById("checkBtn");
  const resultDiv = document.getElementById("result");

  // Load saved API URL
  chrome.storage.local.get(["apiUrl"], (result) => {
    apiUrlInput.value = result.apiUrl || DEFAULT_API_URL;
  });

  saveConfigBtn.addEventListener("click", () => {
    const newUrl = apiUrlInput.value.trim() || DEFAULT_API_URL;
    chrome.storage.local.set({ apiUrl: newUrl }, () => {
      alert("Đã lưu địa chỉ API: " + newUrl);
    });
  });

  // Lấy văn bản đang chọn trên trang
  getSelectionBtn.addEventListener("click", async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) return;

      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => window.getSelection().toString()
      });

      if (results && results[0] && results[0].result) {
        const selectedText = results[0].result.trim();
        if (selectedText.length > 0) {
          newsText.value = selectedText;
        } else {
          alert("Không có đoạn văn bản nào đang được bôi đen/chọn trên trang.");
        }
      } else {
        alert("Không thể trích xuất văn bản từ trang hiện tại.");
      }
    } catch (err) {
      alert("Lỗi khi lấy văn bản: " + err.message);
    }
  });

  // Kiểm tra tin tức
  checkBtn.addEventListener("click", async () => {
    const text = newsText.value.trim();
    if (text.length < 20) {
      alert("Vui lòng nhập hoặc chọn ít nhất 20 ký tự.");
      return;
    }

    const apiUrl = apiUrlInput.value.trim() || DEFAULT_API_URL;

    resultDiv.style.display = "block";
    resultDiv.className = "";
    resultDiv.innerHTML = "⏳ Đang phân tích nội dung...";

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      });

      const data = await response.json();
      if (response.ok) {
        const confPct = (data.confidence_score * 100).toFixed(1);
        const fakeProbPct = (data.fake_probability * 100).toFixed(1);
        const barColor = data.is_fake ? "#dc3545" : "#28a745";

        if (data.is_fake) {
          resultDiv.className = "fake";
          resultDiv.innerHTML = `
            <strong>⚠️ CẢNH BÁO TIN GIẢ!</strong><br>
            Xác suất giả: ${fakeProbPct}% | Độ tự tin: ${confPct}%
            <div class="progress-bar-container">
              <div class="progress-bar" style="width: ${fakeProbPct}%; background-color: ${barColor};"></div>
            </div>
          `;
        } else {
          resultDiv.className = "real";
          resultDiv.innerHTML = `
            <strong>✅ TIN CẬY (TIN THẬT)</strong><br>
            Xác suất giả: ${fakeProbPct}% | Độ tự tin: ${confPct}%
            <div class="progress-bar-container">
              <div class="progress-bar" style="width: ${fakeProbPct}%; background-color: ${barColor};"></div>
            </div>
          `;
        }
      } else {
        resultDiv.className = "fake";
        resultDiv.innerText = "Lỗi API: " + (data.error || "Không rõ nguyên nhân.");
      }
    } catch (err) {
      resultDiv.className = "fake";
      resultDiv.innerHTML = `Lỗi kết nối tới Flask API (${apiUrl}).<br><strong>Hãy chạy: python -m app.app</strong>`;
    }
  });
});
