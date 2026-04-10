document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("submit-btn");
    const loader = document.getElementById("loader");
    const btnText = document.querySelector(".btn-text");
    const errorText = document.getElementById("error-message");
    
    const idleUI = document.getElementById("idle-ui");
    const loadingUI = document.getElementById("loading-ui");
    const resultUI = document.getElementById("result-ui");
    
    const ticketId = document.getElementById("ticket-id");
    const badgeCategory = document.getElementById("badge-category");
    const badgeRisk = document.getElementById("badge-risk");
    const badgeAction = document.getElementById("badge-action");
    const rationaleText = document.getElementById("rationale-text");
    
    const jsonViewer = document.getElementById("json-viewer");
    const rewardSection = document.getElementById("reward-section");
    const rewardStatus = document.getElementById("reward-status");

    let currentEpisodeId = null;
    let typeWriterInterval = null;

    const setCardState = (state) => {
        idleUI.style.display = "none";
        loadingUI.style.display = "none";
        resultUI.style.display = "none";
        
        if (state === "idle") idleUI.style.display = "flex";
        if (state === "loading") loadingUI.style.display = "block";
        if (state === "result") resultUI.style.display = "flex";
    };

    const typeWriter = (text, element, speed = 15) => {
        clearInterval(typeWriterInterval);
        element.innerHTML = '<span class="cursor"></span>';
        let i = 0;
        
        typeWriterInterval = setInterval(() => {
            if (i < text.length) {
                element.innerHTML = text.substring(0, i + 1) + '<span class="cursor"></span>';
                i++;
            } else {
                clearInterval(typeWriterInterval);
                element.innerHTML = text; // Remove cursor after done
            }
        }, speed);
    };

    btn.addEventListener("click", async () => {
        const prompt = document.getElementById("prompt").value;
        const assistantResponse = document.getElementById("assistant-response").value;
        const context = document.getElementById("context").value;
        const mode = document.querySelector('input[name="mode"]:checked').value;

        // Visual loading state
        errorText.textContent = "";
        btnText.textContent = "Processing...";
        loader.style.display = "block";
        btn.disabled = true;
        
        setCardState("loading");
        if(rewardSection) rewardSection.style.display = "none";
        rewardStatus.textContent = "";
        
        jsonViewer.innerHTML = "> Executing inference engine...\n> Requesting /api/infer";

        try {
            const res = await fetch("/api/infer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode, prompt, assistant_response: assistantResponse, context })
            });

            if (!res.ok) throw new Error("Server responded with status " + res.status);

            const data = await res.json();
            
            // Format JSON
            jsonViewer.innerHTML = `<span style="color:var(--accent-primary)">> Payload received:</span>\n${JSON.stringify(data, null, 2)}`;

            currentEpisodeId = data.episode_id;
            
            // Build Ticket View
            ticketId.textContent = `EP-${data.episode_id.split("-")[0].toUpperCase()}`;
            
            badgeCategory.textContent = data.category || "Unknown";
            
            const risk = data.risk_label || "Unknown";
            badgeRisk.textContent = risk.toUpperCase();
            badgeRisk.className = "badge"; // reset
            if (risk === "harmful") badgeRisk.classList.add("badge-risk-harmful");
            else if (risk === "suspicious") badgeRisk.classList.add("badge-risk-suspicious");
            else badgeRisk.classList.add("badge-risk-benign");
            
            badgeAction.textContent = `ACT: ${(data.action || "NONE").toUpperCase()}`;
            
            setCardState("result");
            typeWriter(data.rationale || "No rationale provided by model.", rationaleText);

            if (mode === "training" && currentEpisodeId && rewardSection) {
                // Show reward layout
                rewardSection.style.display = "flex";
                document.getElementById("reward-pos").disabled = false;
                document.getElementById("reward-neg").disabled = false;
            }

        } catch (err) {
            errorText.textContent = "Connection error: " + err.message;
            setCardState("idle");
            jsonViewer.innerHTML = `<span style="color:var(--danger)">> ERROR: ${err.message}</span>`;
        } finally {
            btnText.textContent = "Execute Policy Analysis";
            loader.style.display = "none";
            btn.disabled = false;
        }
    });

    const sendReward = async (rewardVal) => {
        if (!currentEpisodeId) return;
        
        try {
            rewardStatus.textContent = "Synchronizing policy...";
            rewardStatus.style.color = "var(--text-muted)";
            
            const res = await fetch("/api/reward", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ episode_id: currentEpisodeId, reward: rewardVal })
            });
            const data = await res.json();
            
            if (data.status === "success") {
                rewardStatus.textContent = "Policy successfully updated.";
                rewardStatus.style.color = "var(--success)";
                document.getElementById("reward-pos").disabled = true;
                document.getElementById("reward-neg").disabled = true;
            } else {
                rewardStatus.textContent = "Error: " + data.message;
                rewardStatus.style.color = "var(--danger)";
            }
        } catch (err) {
            rewardStatus.textContent = "Terminal Error: Failed to synchronize.";
            rewardStatus.style.color = "var(--danger)";
        }
    };

    const posBtn = document.getElementById("reward-pos");
    const negBtn = document.getElementById("reward-neg");
    if(posBtn) posBtn.addEventListener("click", () => sendReward(1.0));
    if(negBtn) negBtn.addEventListener("click", () => sendReward(-1.0));
});
