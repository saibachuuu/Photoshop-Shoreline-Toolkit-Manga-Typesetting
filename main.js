/* main.js - UXP Photoshop Plugin Entry */

const { entrypoints } = require("uxp");
const { loadTranslations, applyTranslations, t } = require("./i18n/i18n");
const { handleImageOutput } = require("./features/imageOutput");
const { handleBatchOutput } = require("./features/batchOutput");
const { handleMojiRemove, handleApplyResult, setSelectedModel, handleForceStop } = require("./features/mojiRemove");
const { loadSettings, saveSettings, getSettings } = require("./config");

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", async () => {
  // Load translations from JSON files
  await loadTranslations();
  
  // Apply translations to UI
  applyTranslations();
  
  // Load saved settings
  await loadSettings();
  
  // Attach event listeners
  const btnImageOutput = document.getElementById("btnImageOutput");
  btnImageOutput.addEventListener("click", handleImageOutput);

  const btnBatchOutput = document.getElementById("btnBatchOutput");
  btnBatchOutput.addEventListener("click", handleBatchOutput);
  
  const btnMojiRemove = document.getElementById("btnMojiRemove");
  btnMojiRemove.addEventListener("click", handleMojiRemove);

  const btnApplyResult = document.getElementById("btnApplyResult");
  btnApplyResult.addEventListener("click", handleApplyResult);

  // Force Stop button
  const btnForceStop = document.getElementById("btnForceStop");
  btnForceStop.addEventListener("click", handleForceStop);

  // Settings button
  const btnSettings = document.getElementById("btnSettings");
  btnSettings.addEventListener("click", openSettingsDialog);

  // Model selector
  const modelSelector = document.getElementById("modelSelector");
  modelSelector.addEventListener("change", (evt) => {
    setSelectedModel(evt.target.value);
  });
  
  // Initialize with saved model or default
  const savedModel = localStorage.getItem("selectedModel") || "gemini";
  const radios = modelSelector.querySelectorAll("sp-radio");
  radios.forEach(radio => {
    if (radio.value === savedModel) {
      radio.checked = true;
    }
  });
  setSelectedModel(savedModel);
});

/**
 * Open settings dialog
 */
async function openSettingsDialog() {
  const dialog = document.getElementById("settingsDialog");
  const settings = getSettings();
  
  // Populate current settings with defaults
  document.getElementById("sharedPrompt").value = settings.prompt || "Remove the stylized Japanese text and fill the black border.";
  document.getElementById("geminiEndpoint").value = settings.geminiEndpoint || "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent";
  document.getElementById("geminiApiKey").value = settings.geminiApiKey || "";
  document.getElementById("fluxEndpoint").value = settings.fluxEndpoint || "https://api.together.xyz/v1/images/generations";
  document.getElementById("fluxApiKey").value = settings.fluxApiKey || "";
  document.getElementById("fluxSteps").value = settings.fluxSteps || "20";
  document.getElementById("fluxGuidanceScale").value = settings.fluxGuidanceScale || "3.0";

  // Setup button handlers
  const btnSave = document.getElementById("btnSaveSettings");
  const btnCancel = document.getElementById("btnCancelSettings");
  
  btnSave.onclick = () => {
    const newSettings = {
      prompt: document.getElementById("sharedPrompt").value,
      geminiEndpoint: document.getElementById("geminiEndpoint").value,
      geminiApiKey: document.getElementById("geminiApiKey").value,
      fluxEndpoint: document.getElementById("fluxEndpoint").value,
      fluxApiKey: document.getElementById("fluxApiKey").value,
      fluxSteps: document.getElementById("fluxSteps").value,
      fluxGuidanceScale: document.getElementById("fluxGuidanceScale").value
    };
    
    saveSettings(newSettings);
    dialog.close("saved");
  };
  
  btnCancel.onclick = () => {
    dialog.close("cancelled");
  };
  
  // Show dialog
  try {
    const result = await dialog.uxpShowModal({
      title: t("settingsTitle") || "API Settings",
      resize: "both",
      size: {
        width: 480,
        height: 800
      }
    });
    
    console.log("Settings dialog result:", result);
  } catch (err) {
    console.error("Error showing settings dialog:", err);
  }
}
