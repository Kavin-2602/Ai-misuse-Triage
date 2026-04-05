document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("submit-btn");
    const spinner = document.getElementById("spinner");
    const btnText = document.querySelector(".btn-text");
    const errorText = document.getElementById("error-message");
    const jsonViewer = document.getElementById("json-viewer");
    const statusRing = document.querySelector(".status-ring");
    const statusLbl = document.querySelector(".status-lbl");

    let currentEpisodeId = null;

    const syntaxHighlight = (json) => {
        if (typeof json != 'string') {
            json = JSON.stringify(json, undefined, 2);
        }
        json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                } else {
                    cls = 'json-string';
                    // Special styling for risk labels if they appear as values
                    if (match === '"harmful"') cls += ' risk-harmful';
                    if (match === '"suspicious"') cls += ' risk-suspicious';
                    if (match === '"benign"') cls += ' risk-benign';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-boolean';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
            return '<span class="' + cls + '">' + match + '</span>';
        });
    };

    btn.addEventListener("click", async () => {
        const prompt = document.getElementById("prompt").value;
        const assistantResponse = document.getElementById("assistant-response").value;
        const context = document.getElementById("context").value;
        const modeInput = document.querySelector('input[name="mode"]:checked');
        const mode = modeInput ? modeInput.value : "evaluation";

        // Visual loading state
        errorText.textContent = "";
        btnText.textContent = "Running Analysis...";
        spinner.style.display = "block";
        statusRing.style.backgroundColor = "var(--warn)";
        statusRing.style.boxShadow = "0 0 8px var(--warn)";
        statusLbl.textContent = "Analyzing input telemetry...";
        
        // Hide reward section while inferring
        const rewardSection = document.getElementById("reward-section");
        const rewardStatus = document.getElementById("reward-status");
        if(rewardSection) rewardSection.style.display = "none";

        try {
            const res = await fetch("/api/infer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode, prompt, assistant_response: assistantResponse, context })
            });

            if (!res.ok) {
                throw new Error("Server responded with status " + res.status);
            }

            const data = await res.json();
            
            // Render JSON nicely format 
            jsonViewer.innerHTML = syntaxHighlight(data);

            currentEpisodeId = data.episode_id;
            if (mode === "training" && currentEpisodeId && rewardSection) {
                rewardSection.style.display = "block";
                rewardStatus.textContent = "";
            }

            // Update status indicator
            const risk = data.risk_label || "unknown";
            if (risk === "harmful") {
                statusRing.style.backgroundColor = "var(--danger)";
                statusRing.style.boxShadow = "0 0 10px var(--danger)";
                statusLbl.textContent = "Priority Alert: Harmful";
            } else if (risk === "suspicious") {
                statusRing.style.backgroundColor = "var(--warn)";
                statusRing.style.boxShadow = "0 0 10px var(--warn)";
                statusLbl.textContent = "Caution: Suspicious";
            } else {
                statusRing.style.backgroundColor = "var(--success)";
                statusRing.style.boxShadow = "0 0 10px var(--success)";
                statusLbl.textContent = "Clear: Benign";
            }

        } catch (err) {
            errorText.textContent = "Connection error: " + err.message;
            statusRing.style.backgroundColor = "var(--text-muted)";
            statusRing.style.boxShadow = "none";
            statusLbl.textContent = "Execution Failed";
        } finally {
            btnText.textContent = "Execute Inference Evaluation";
            spinner.style.display = "none";
        }
    });

    const sendReward = async (rewardVal) => {
        if (!currentEpisodeId) return;
        const rewardStatus = document.getElementById("reward-status");
        
        try {
            rewardStatus.textContent = "Sending feedback...";
            rewardStatus.style.color = "var(--text-muted)";
            
            const res = await fetch("/api/reward", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ episode_id: currentEpisodeId, reward: rewardVal })
            });
            const data = await res.json();
            
            if (data.status === "success") {
                rewardStatus.textContent = data.message;
                rewardStatus.style.color = "var(--success)";
                document.getElementById("reward-pos").disabled = true;
                document.getElementById("reward-neg").disabled = true;
                
                // Keep the success state briefly, we don't hide it so the user knows it worked.
            } else {
                rewardStatus.textContent = "Error: " + data.message;
                rewardStatus.style.color = "var(--danger)";
            }
        } catch (err) {
            rewardStatus.textContent = "Failed to send reward parameters.";
            rewardStatus.style.color = "var(--danger)";
        }
    };

    const posBtn = document.getElementById("reward-pos");
    const negBtn = document.getElementById("reward-neg");
    if(posBtn) posBtn.addEventListener("click", () => sendReward(1.0));
    if(negBtn) negBtn.addEventListener("click", () => sendReward(-1.0));
});
