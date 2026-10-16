// ============================================================
// PHISH//X WEB APP
// Frontend interaction
// ============================================================


// ============================================================
// PAGE LOAD
// ============================================================

document.addEventListener("DOMContentLoaded", function () {

    console.log("PHISH//X interface initialized.");

    setupScanner();

    setupHtmlCounter();

    setupUrlInput();

    animateExistingResult();

});


// ============================================================
// SCANNER SETUP
// ============================================================

function setupScanner() {

    const form = document.querySelector(
        'form[action="/scan"]'
    );

    if (!form) {
        return;
    }

    form.addEventListener(
        "submit",
        function () {

            const button = form.querySelector(
                'button[type="submit"]'
            );

            const urlInput = form.querySelector(
                'input[name="url"]'
            );

            if (!urlInput.value.trim()) {
                return;
            }


            // Prevent double clicking
            if (form.dataset.scanning === "true") {
                return;
            }

            form.dataset.scanning = "true";


            // Change button appearance
            if (button) {

                button.disabled = true;

                button.innerHTML =
                    `
                    <span class="scan-spinner"></span>
                    ANALYZING...
                    `;

            }


            // Show scan status
            showScanStatus();

        }
    );

}


// ============================================================
// SCAN STATUS
// ============================================================

function showScanStatus() {

    // Don't create it twice
    if (
        document.querySelector(
            ".scan-status"
        )
    ) {
        return;
    }


    const scanner = document.querySelector(
        ".scanner-card"
    );

    if (!scanner) {
        return;
    }


    const status = document.createElement(
        "div"
    );

    status.className = "scan-status";

    status.innerHTML =
        `
        <div class="terminal-line">
            <span class="terminal-symbol">></span>
            Initializing feature extraction...
        </div>

        <div class="terminal-line">
            <span class="terminal-symbol">></span>
            Running phishing classifier...
        </div>

        <div class="terminal-line">
            <span class="terminal-symbol">></span>
            Generating evidence...
        </div>
        `;


    scanner.appendChild(status);

}


// ============================================================
// HTML CHARACTER COUNTER
// ============================================================

function setupHtmlCounter() {

    const textarea = document.querySelector(
        'textarea[name="html"]'
    );

    if (!textarea) {
        return;
    }


    const counter = document.createElement(
        "div"
    );

    counter.className = "html-counter";

    counter.textContent = "0 characters";

    textarea.parentNode.appendChild(
        counter
    );


    textarea.addEventListener(
        "input",
        function () {

            const length =
                textarea.value.length;


            counter.textContent =
                formatNumber(length) +
                " characters";


            if (length > 0) {

                counter.classList.add(
                    "active"
                );

            } else {

                counter.classList.remove(
                    "active"
                );

            }

        }
    );

}


// ============================================================
// URL INPUT
// ============================================================

function setupUrlInput() {

    const input = document.querySelector(
        'input[name="url"]'
    );

    if (!input) {
        return;
    }


    input.addEventListener(
        "input",
        function () {

            const value =
                input.value.trim();


            if (
                value.length > 0
            ) {

                input.classList.add(
                    "has-value"
                );

            } else {

                input.classList.remove(
                    "has-value"
                );

            }

        }
    );

}


// ============================================================
// RESULT ANIMATION
// ============================================================

function animateExistingResult() {

    const result =
        document.querySelector(
            ".result-section"
        );

    if (!result) {
        return;
    }


    result.classList.add(
        "result-enter"
    );


    // Animate evidence cards
    const cards =
        document.querySelectorAll(
            ".evidence-card"
        );


    cards.forEach(
        function (card, index) {

            card.style.animationDelay =
                (index * 80) + "ms";

            card.classList.add(
                "evidence-enter"
            );

        }
    );


    // Animate feature cards
    const features =
        document.querySelectorAll(
            ".feature-card"
        );


    features.forEach(
        function (card, index) {

            card.style.animationDelay =
                (index * 60) + "ms";

            card.classList.add(
                "feature-enter"
            );

        }
    );


    animateRiskScore();

}


// ============================================================
// RISK SCORE ANIMATION
// ============================================================

function animateRiskScore() {

    const number =
        document.querySelector(
            ".risk-number"
        );

    if (!number) {
        return;
    }


    const target =
        parseFloat(
            number.textContent
        );


    if (isNaN(target)) {
        return;
    }


    let current = 0;

    const duration = 900;

    const start =
        performance.now();


    function update(timestamp) {

        const elapsed =
            timestamp - start;


        const progress =
            Math.min(
                elapsed / duration,
                1
            );


        // Smooth easing
        const eased =
            1 -
            Math.pow(
                1 - progress,
                3
            );


        current =
            target * eased;


        number.textContent =
            current.toFixed(1);


        if (progress < 1) {

            requestAnimationFrame(
                update
            );

        } else {

            number.textContent =
                target.toFixed(1);

        }

    }


    requestAnimationFrame(
        update
    );

}


// ============================================================
// NUMBER FORMATTER
// ============================================================

function formatNumber(number) {

    return number.toLocaleString(
        "en-US"
    );

}


// ============================================================
// COPY URL
// ============================================================

function addCopyButton() {

    const scannedUrl =
        document.querySelector(
            ".scanned-url"
        );

    if (!scannedUrl) {
        return;
    }


    const url =
        scannedUrl.textContent.trim();


    if (!url) {
        return;
    }


    const button =
        document.createElement(
            "button"
        );


    button.className =
        "copy-url-button";

    button.textContent =
        "COPY";


    button.addEventListener(
        "click",
        async function () {

            try {

                await navigator.clipboard.writeText(
                    url
                );

                button.textContent =
                    "COPIED";

                setTimeout(
                    function () {

                        button.textContent =
                            "COPY";

                    },
                    1500
                );

            } catch (error) {

                console.log(
                    "Could not copy URL."
                );

            }

        }
    );


    scannedUrl.parentNode.appendChild(
        button
    );

}


// ============================================================
// INITIALIZE COPY BUTTON
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        addCopyButton();

    }
);