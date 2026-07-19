// ================================================================================
// 生成プロンプトをクリップボードへコピーする UI 補助スクリプト
// ================================================================================
(function () {
    // ================================================================================
    // 生成プロンプト欄のテキストを取得する
    // ================================================================================
    function getPromptText() {
        const root = document.getElementById("llm_prompt_ollama_generated_prompt");
        if (!root) return "";
        const ta = root.querySelector("textarea");
        return ta ? ta.value : "";
    }

    // ================================================================================
    // Copy ボタンにクリップボード書き込みを紐付ける
    // ================================================================================
    function bind() {
        const btn = document.getElementById("llm_prompt_ollama_copy_prompt_btn");
        if (!btn || btn.dataset.llmPromptOllamaCopyBound === "1") return;
        btn.dataset.llmPromptOllamaCopyBound = "1";
        btn.addEventListener("click", async function (ev) {
            // Let Gradio handle its own click too; we just copy.
            const text = getPromptText();
            if (!text) return;
            try {
                await navigator.clipboard.writeText(text);
            } catch (err) {
                // Fallback for older browsers / insecure context
                const ta = document.querySelector("#llm_prompt_ollama_generated_prompt textarea");
                if (!ta) return;
                ta.focus();
                ta.select();
                document.execCommand("copy");
            }
        });
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(bind);
    } else {
        document.addEventListener("DOMContentLoaded", bind);
        setTimeout(bind, 1500);
    }
})();
