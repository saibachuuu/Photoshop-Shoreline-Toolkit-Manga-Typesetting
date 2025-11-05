/* mojiRemove.js - Text removal feature using Gemini or FLUX Kontext */

const { t, updateStatus } = require("../i18n/i18n");
const { getSettings } = require("../config");
const photoshop = require("photoshop");
const { app, core } = photoshop;
const { batchPlay } = require("photoshop").action;

const uxp = require("uxp");
const fs = uxp.storage.localFileSystem;
const { formats } = uxp.storage;

// Store processing state
let processingState = {
  documentName: null,
  documentId: null,
  selectionBounds: null,
  resultBase64: null,
  width: null,
  height: null,
  resolution: null,
  isProcessing: false,
  abortController: null
};

// Current selected model
let selectedModel = "gemini";

/**
 * Set the selected model
 */
function setSelectedModel(model) {
  selectedModel = model;
  window.localStorage.setItem("selectedModel", model);
  console.log("Model selected:", model);
}

/**
 * Enable or disable buttons
 */
function setApplyButtonState(enabled) {
  const btnApply = document.getElementById("btnApplyResult");
  if (btnApply) {
    btnApply.disabled = !enabled;
  }
}

function setForceStopButtonState(enabled) {
  const btnForceStop = document.getElementById("btnForceStop");
  if (btnForceStop) {
    btnForceStop.disabled = !enabled;
  }
}

/**
 * Convert selection to base64 PNG
 */
async function getSelectionAsBase64() {
  const doc = app.activeDocument;
  
  if (!doc.selection.bounds) {
    throw new Error(t("noSelection"));
  }
  const selectionBounds = doc.selection.bounds;

  const bounds = {
    left: Math.floor(selectionBounds.left),
    top: Math.floor(selectionBounds.top),
    right: Math.ceil(selectionBounds.right),
    bottom: Math.ceil(selectionBounds.bottom)
  };

  const width = bounds.right - bounds.left;
  const height = bounds.bottom - bounds.top;

  updateStatus(t("processingSelection").replace("{width}", width).replace("{height}", height));

  const hasOnlyLockedBackground = doc.layers.length === 1;

  let tempDoc;
  
  if (hasOnlyLockedBackground) {
  // Copy the selection
  await batchPlay([
    {
      _obj: "copyEvent",
      _options: { dialogOptions: "dontDisplay" }
    }
  ], {});

  // Create a new temporary document using app.createDocument
  tempDoc = await app.createDocument({
    width: width,
    height: height,
    resolution: doc.resolution,
    mode: "RGBColorMode",
    fill: "transparent"
  });

  // Paste the selection
  await batchPlay([
    {
      _obj: "paste",
      _options: { dialogOptions: "dontDisplay" }
    }
  ], {});
    
  } else {

    await batchPlay([{ _obj: "copyMerged", _options: { dialogOptions: "dontDisplay" } }], {});

    tempDoc = await app.createDocument({
      width: width,
      height: height,
      resolution: doc.resolution,
      mode: "RGBColorMode",
      fill: "transparent"
    });

    await tempDoc.paste();
  }
  
  await tempDoc.flatten();

  const tempFolder = await fs.getTemporaryFolder();
  const tempFile = await tempFolder.createFile("temp_selection.png", { overwrite: true });

  await tempDoc.saveAs.png(tempFile, { compression: 6 }, true);
  const arrayBuffer = await tempFile.read({ format: formats.binary });
  const base64 = arrayBufferToBase64(arrayBuffer);

  await tempDoc.closeWithoutSaving();
  await tempFile.delete();
  
  app.activeDocument = doc;

  return { base64, width, height, resolution: doc.resolution, bounds };
}

/**
 * Convert ArrayBuffer to base64 string
 */
function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Send image to Gemini API
 */
async function sendToGemini(base64Image) {
  const settings = getSettings();
  const prompt = settings.prompt || "Remove the stylized Japanese text and fill the black border.";
  
  if (!settings.geminiEndpoint || !settings.geminiApiKey) {
    throw new Error(t("apiNotConfigured").replace("{modelName}", "Gemini"));
  }

  const requestBody = {
    contents: [{
      parts: [
        { inline_data: { mime_type: "image/png", data: base64Image } },
        { text: prompt }
      ]
    }]
  };

  // Create abort controller
  processingState.abortController = new AbortController();

  const response = await fetch(settings.geminiEndpoint, {
    method: "POST",
    headers: {
      "x-goog-api-key": settings.geminiApiKey,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(requestBody),
    signal: processingState.abortController.signal
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini API Error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  const part = data.candidates?.[0]?.content?.parts?.find(p => p.inlineData?.data);
  if (part) {
    return part.inlineData.data;
  }
  throw new Error(t("noImageData"));
}

/**
 * Round dimension to nearest multiple of 8
 */
function roundToMultipleOf8(value) {
  return Math.round(value / 8) * 8;
}

/**
 * Send image to FLUX API
 */
async function sendToFlux(base64Image, width, height) {
  const settings = getSettings();
  
  if (!settings.fluxEndpoint || !settings.fluxApiKey) {
    throw new Error(t("apiNotConfigured").replace("{modelName}", "FLUX"));
  }
  
  const prompt = settings.prompt || "Remove the stylized Japanese text, extend the canvas to fill the gray area.";
  const steps = parseInt(settings.fluxSteps, 10) || 20;
  const guidance_scale = parseFloat(settings.fluxGuidanceScale) || 3.0;

  const imageUrl = `data:image/png;base64,${base64Image}`;
  const validWidth = roundToMultipleOf8(Math.min(Math.max(width, 256), 1440));
  const validHeight = roundToMultipleOf8(Math.min(Math.max(height, 256), 1440));

  const requestBody = {
    model: "black-forest-labs/FLUX.1-kontext-dev",
    prompt: prompt,
    n: 1,
    width: validWidth,
    height: validHeight,
    steps: steps,
    guidance_scale: guidance_scale,
    disable_safety_checker: true,
    image_url: imageUrl
  };

  processingState.abortController = new AbortController();

  const response = await fetch(settings.fluxEndpoint, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${settings.fluxApiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(requestBody),
    signal: processingState.abortController.signal
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`FLUX API Error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  const resultUrl = data.data?.[0]?.url;

  if (resultUrl) {
    const imageResponse = await fetch(resultUrl, {
      signal: processingState.abortController.signal
    });
    if (!imageResponse.ok) throw new Error(`Failed to download FLUX result: ${imageResponse.status}`);
    const arrayBuffer = await imageResponse.arrayBuffer();
    return arrayBufferToBase64(arrayBuffer);
  }
  throw new Error("No image URL in FLUX response");
}

/**
 * Main handler for Moji Remove feature
 */
async function handleMojiRemove() {
  if (processingState.isProcessing) return;

  try {
    processingState.isProcessing = true;
    processingState.abortController = null;
    setApplyButtonState(false);
    setForceStopButtonState(false);
    updateStatus(t("processing"));

    const selectionData = await core.executeAsModal(async () => {
      const doc = app.activeDocument;
      updateStatus(t("readingSelection"));
      const { base64, width, height, resolution, bounds } = await getSelectionAsBase64();
      return { docName: doc.name, docId: doc.id, base64, width, height, resolution, bounds };
    }, { commandName: "Moji Remove - Capture" });

    processingState.documentName = selectionData.docName;
    processingState.documentId = selectionData.docId;
    processingState.resultBase64 = null;
    processingState.selectionBounds = selectionData.bounds;
    processingState.width = selectionData.width;
    processingState.height = selectionData.height;
    processingState.resolution = selectionData.resolution;

    const modelName = selectedModel === "gemini" ? "Gemini" : "FLUX";
    updateStatus(t("processingWithModel").replace("{modelName}", modelName).replace("{docName}", selectionData.docName));
    
    setForceStopButtonState(true);
    
    const resultBase64 = await (selectedModel === "gemini" 
      ? sendToGemini(selectionData.base64)
      : sendToFlux(selectionData.base64, selectionData.width, selectionData.height));
      
    processingState.resultBase64 = resultBase64;
    setApplyButtonState(true);
    updateStatus(t("resultReady").replace("{docName}", processingState.documentName));

  } catch (error) {
    if (error.name === 'AbortError') {
      console.log("Request aborted by user");
      updateStatus(t("requestAborted"));
    } else {
      console.error("Moji Remove error:", error);
      updateStatus(`${t("error")}: ${error.message}`);
      await core.showAlert({ message: `${t("mojiRemoveFailed")}: ${error.message}` });
    }
  } finally {
    processingState.isProcessing = false;
    processingState.abortController = null;
    setForceStopButtonState(false);
  }
}

/**
 * Force stop handler
 */
function handleForceStop() {
  if (processingState.abortController) {
    processingState.abortController.abort();
    updateStatus(t("stoppingRequest"));
  }
}

/**
 * Handler for Apply Result button
 */
async function handleApplyResult() {
  if (!processingState.resultBase64 || !processingState.documentId) {
    return;
  }

  try {
    const currentDoc = app.activeDocument;
    
    if (currentDoc.id !== processingState.documentId) {
      await core.showAlert({
        message: `Please open document: ${processingState.documentName}\nCurrent document: ${currentDoc.name}`
      });
      return;
    }

    updateStatus("Applying result...");

    await core.executeAsModal(async () => {
      const targetDoc = app.activeDocument;
      const targetDocName = targetDoc.name;
      
      // Deselect if there's an active selection
      try {
        const selectionBounds = targetDoc.selection.bounds;
        if (selectionBounds) {
          await batchPlay([
            {
              _obj: "set",
              _target: [
                {
                  _ref: "channel",
                  _property: "selection"
                }
              ],
              to: {
                _enum: "ordinal",
                _value: "none"
              },
              _options: { dialogOptions: "dontDisplay" }
            }
          ], {});
        }
      } catch (e) {
        console.log("No selection to deselect:", e);
      }
      
      // Write result image to temporary file
      const tempFolder = await fs.getTemporaryFolder();
      const resultFile = await tempFolder.createFile("moji_result.png", { overwrite: true });

      const binaryString = atob(processingState.resultBase64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      await resultFile.write(bytes.buffer, { format: formats.binary });

      // Open result as temporary document
      const tempDoc = await app.open(resultFile);
      await tempDoc.flatten();
      
      const sourceLayer = tempDoc.layers[0];
      
      // Unlock layer if needed
      if (sourceLayer.allLocked) sourceLayer.allLocked = false;
      if (sourceLayer.pixelsLocked) sourceLayer.pixelsLocked = false;
      if (sourceLayer.positionLocked) sourceLayer.positionLocked = false;
      if (sourceLayer.transparentPixelsLocked) sourceLayer.transparentPixelsLocked = false;
      
      tempDoc.activeLayers = [sourceLayer];
      
      // Duplicate layer to target document
      await batchPlay([
        {
          _obj: "duplicate",
          _target: [
            {
              _ref: "layer",
              _enum: "ordinal",
              _value: "targetEnum"
            }
          ],
          to: {
            _ref: "document",
            _name: targetDocName
          },
          name: "Moji Removed",
          _options: { dialogOptions: "dontDisplay" }
        }
      ], {});
      
      await tempDoc.closeWithoutSaving();
      await resultFile.delete();

      // Switch back to target document
      app.activeDocument = targetDoc;
      
      const newLayer = targetDoc.activeLayers[0];
      
      // Position the layer at the original selection bounds
      const layerBounds = newLayer.bounds;
      const currentLeft = layerBounds.left;
      const currentTop = layerBounds.top;
      
      const targetLeft = processingState.selectionBounds.left;
      const targetTop = processingState.selectionBounds.top;
      
      const offsetX = targetLeft - currentLeft;
      const offsetY = targetTop - currentTop;
      
      if (offsetX !== 0 || offsetY !== 0) {
        await newLayer.translate(offsetX, offsetY);
      }

      // Make compare comfortable
      newLayer.visible = false;
      const dummyLayer = await targetDoc.createLayer({ name: "_temp_" });
      await dummyLayer.delete();
      newLayer.visible = true;

    }, { commandName: "Apply Moji Remove Result" });

    // Clear processing state
    processingState.resultBase64 = null;
    processingState.documentName = null;
    processingState.documentId = null;
    processingState.selectionBounds = null;
    setApplyButtonState(false);
    
    updateStatus(t("mojiRemoveComplete"));

  } catch (error) {
    console.error("Apply result error:", error);
    updateStatus(`${t("error")}: ${error.message}`);
    
    await core.showAlert({
      message: `Failed to apply result: ${error.message}`
    });
  }
}

module.exports = { handleMojiRemove, handleApplyResult, setSelectedModel, handleForceStop };
