/* i18n.js - Multilingual support with JSON file loading */

const uxp = require("uxp");
const fs = uxp.storage.localFileSystem;

let currentTranslations = {};

// Map locale variants to base locale
function getLocaleKey(locale) {
  if (locale.startsWith("zh")) return "zh_CN";
  if (locale.startsWith("ko")) return "ko_KR";
  return "en_US";  // Default to English
}

// Load translations from JSON file
async function loadTranslations() {
  try {
    const currentLocale = uxp.host.uiLocale;
    const localeKey = getLocaleKey(currentLocale);
    
    // Get plugin folder
    const pluginFolder = await fs.getPluginFolder();
    const localesFolder = await pluginFolder.getEntry("locales");
    const localeFile = await localesFolder.getEntry(`${localeKey}.json`);
    
    // Read and parse JSON
    const content = await localeFile.read();
    currentTranslations = JSON.parse(content);
    
    console.log(`Loaded translations for: ${localeKey}`);
    return currentTranslations;
  } catch (e) {
    console.error("Failed to load translations:", e);
    // Fallback to English if loading fails
    currentTranslations = await loadFallbackTranslations();
    return currentTranslations;
  }
}

// Fallback: load English translations
async function loadFallbackTranslations() {
  try {
    const pluginFolder = await fs.getPluginFolder();
    const localesFolder = await pluginFolder.getEntry("locales");
    const localeFile = await localesFolder.getEntry("en_US.json");
    const content = await localeFile.read();
    return JSON.parse(content);
  } catch (e) {
    console.error("Failed to load fallback translations:", e);
    return {};
  }
}

// Get translation by key
function t(key) {
  return currentTranslations[key] || key;
}

// Apply translations to DOM elements
function applyTranslations() {
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    const translation = t(key);
    if (translation) {
      el.textContent = translation;
    }
  });
  // Special handler for sp-radio labels
  document.querySelectorAll("[data-i18n-label]").forEach(el => {
    const key = el.getAttribute("data-i18n-label");
    const translation = t(key);
    if (translation) {
      const textNode = document.createTextNode(translation);
      // Clear existing content and append new label
      while (el.firstChild) {
          el.removeChild(el.firstChild);
      }
      el.appendChild(textNode);
    }
  });
}

// Update status bar helper
function updateStatus(message) {
  const statusBar = document.getElementById("statusBar");
  if (statusBar) {
    statusBar.textContent = message;
  }
}

module.exports = { 
  loadTranslations, 
  t, 
  applyTranslations,
  updateStatus
};
