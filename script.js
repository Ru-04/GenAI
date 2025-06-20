// Optional: Detect unexpected page reload
window.addEventListener("beforeunload", () => {
  console.warn("[RELOAD DETECTED] Page is reloading...");
});

async function submitPrompt(event) {
  console.log("[submitPrompt] Function triggered.");
  event.preventDefault(); // Prevent default form submission

  const prompt = document.getElementById('promptInput').value.trim();
  const loading = document.getElementById('loading');
  const responseBox = document.getElementById('responseBox');
  const codeOutput = document.getElementById('codeOutput');
  const outputBox = document.getElementById('outputBox');

  if (!prompt) {
    alert("Please enter a prompt.");
    return;
  }

  // Reset UI state
  loading.classList.remove('hidden');
  responseBox.classList.add('hidden');
  codeOutput.textContent = "";
  outputBox.textContent = "";

  try {
    console.log("[submitPrompt] Sending fetch request...");

    const response = await fetch('http://localhost:5001/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const result = await response.json();
    console.log("[submitPrompt] Response received:", result);

    // Show results
    loading.classList.add('hidden');
    responseBox.classList.remove('hidden');

    codeOutput.textContent = result.code || "No code generated.";

    if (result.success) {
      outputBox.textContent = result.output || "(No output printed)";
      console.log("[OUTPUT]:", result.output);
    } else {
      outputBox.textContent = result.error || "(An error occurred)";
      console.error("[ERROR]:", result.error);
    }

  } catch (err) {
    console.error("[submitPrompt] Fetch failed:", err);
    loading.classList.add('hidden');
    responseBox.classList.remove('hidden');
    codeOutput.textContent = "";
    outputBox.textContent = `❌ Fetch Error: ${err.message}`;
  }
}
