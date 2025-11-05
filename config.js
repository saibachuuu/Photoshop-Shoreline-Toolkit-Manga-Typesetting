/* config.js - API Configuration */

// Default configuration
let API_CONFIG = {
  geminiEndpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent",
  geminiApiKey: "",
  fluxEndpoint: "https://api.together.xyz/v1/images/generations",
  fluxApiKey: "",
  fluxSteps: "20",
  fluxGuidanceScale: "2.5",
  prompt: "Remove the stylized Japanese text."
};


/**
 * Load settings from localStorage
 */
function loadSettings() {
  try {
    const saved = window.localStorage.getItem("shorelineSettings");
    if (saved) {
      const parsed = JSON.parse(saved);
      API_CONFIG = { ...API_CONFIG, ...parsed };
      console.log("Settings loaded:", API_CONFIG);
    }
  } catch (err) {
    console.error("Error loading settings:", err);
  }
}

/**
 * Save settings to localStorage
 */
function saveSettings(settings) {
  try {
    API_CONFIG = { ...API_CONFIG, ...settings };
    window.localStorage.setItem("shorelineSettings", JSON.stringify(API_CONFIG));
    console.log("Settings saved:", API_CONFIG);
  } catch (err) {
    console.error("Error saving settings:", err);
  }
}

/**
 * Get current settings
 */
function getSettings() {
  return { ...API_CONFIG };
}

module.exports = { API_CONFIG, loadSettings, saveSettings, getSettings };
