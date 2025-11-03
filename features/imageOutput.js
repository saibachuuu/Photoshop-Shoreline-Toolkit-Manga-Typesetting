/* imageOutput.js - Single image output feature */

const { app, action, imaging } = require("photoshop");
const uxp = require("uxp");
const { t, updateStatus } = require("../i18n/i18n");

// Resize image data to 32x32 using simple downsampling
function resizeImageTo32x32(pixelData, originalWidth, originalHeight) {
  const targetSize = 32;
  const scaleX = originalWidth / targetSize;
  const scaleY = originalHeight / targetSize;
  
  const resizedData = new Uint8ClampedArray(targetSize * targetSize * 4);
  
  for (let y = 0; y < targetSize; y++) {
    for (let x = 0; x < targetSize; x++) {
      const srcX = Math.floor(x * scaleX);
      const srcY = Math.floor(y * scaleY);
      const srcIndex = (srcY * originalWidth + srcX) * 4;
      const dstIndex = (y * targetSize + x) * 4;
      
      resizedData[dstIndex] = pixelData[srcIndex];         // R
      resizedData[dstIndex + 1] = pixelData[srcIndex + 1]; // G
      resizedData[dstIndex + 2] = pixelData[srcIndex + 2]; // B
      resizedData[dstIndex + 3] = pixelData[srcIndex + 3]; // A
    }
  }
  
  return resizedData;
}

// Check if pixel is close to black and white
function isPixelBW(r, g, b) {
  const channels = [r, g, b];
  const max = Math.max(...channels);
  const min = Math.min(...channels);
  return (max - min) <= 8;
}

// Analyze if image is black and white
function analyzeImageBW(pixelData) {
  updateStatus(t("analyzing"));
  
  for (let i = 0; i < pixelData.length; i += 4) {
    const r = pixelData[i];
    const g = pixelData[i + 1];
    const b = pixelData[i + 2];
    
    if (!isPixelBW(r, g, b)) {
      return false; // Color image
    }
  }
  
  return true; // Black and white image
}

// Main image output handler - wrapped in executeAsModal
async function handleImageOutputModal(executionContext) {
  try {
    const { constants } = require("photoshop");
    
    // Check if document is open
    if (!app.activeDocument) {
      updateStatus(t("noDocument"));
      return;
    }
    
    const doc = app.activeDocument;
    const docName = doc.name
      .replace(/\.[^/.]+$/, "")
      .replace(/[\/:*?"<>|#]/g, "_");
    
    // Show save dialog
    const entry = await uxp.storage.localFileSystem.getFileForSaving(
      docName,
      { types: ["xxx", "jpg", "png"] }
    );
    
    if (!entry) {
      updateStatus(t("saveCancelled"));
      return;
    }
    
    const userFileName = entry.name.replace(/\.[^/.]+$/, "");
    
    // Get parent folder
    const nativePath = entry.nativePath;
    const lastSlashIndex = Math.max(nativePath.lastIndexOf('/'), nativePath.lastIndexOf('\\'));
    const parentPath = nativePath.substring(0, lastSlashIndex);
    const parentFolder = await uxp.storage.localFileSystem.getEntryWithUrl(`file:${parentPath}`);
    
    // Delete placeholder file
    try {
      await entry.delete();
    } catch (e) {
      console.warn("Could not delete placeholder file:", e);
    }
    
    const layerCount = doc.layers.length;
    const needsFlattening = layerCount > 1;
    
    let workingDoc;
    let isDuplicate = false;
    
    if (needsFlattening) {
      updateStatus(t("duplicating"));
      workingDoc = await doc.duplicate(`${userFileName}_temp`);
      app.activeDocument = workingDoc;
      isDuplicate = true;
      
      updateStatus(t("flattening"));
      await action.batchPlay([{
        "_obj": "flattenImage",
        "_isCommand": true
      }], {});
    } else {
      updateStatus(t("processing"));
      workingDoc = doc;
      isDuplicate = false;
    }
    
    // Convert indexed color to RGB
    if (workingDoc.mode === constants.DocumentMode.INDEXEDCOLOR) {
      updateStatus(t("convertingToRGB"));
      await workingDoc.changeMode(constants.ChangeMode.RGB);
    }
    
    // Handle already grayscale documents
    if (workingDoc.mode === constants.DocumentMode.GRAYSCALE) {
      updateStatus(t("outputGrayscale"));
      
      const pngFile = await parentFolder.createFile(`${userFileName}.png`, { overwrite: true });
      await workingDoc.saveAs.png(pngFile, { compression: 6 }, true);
      
      updateStatus(`${t("complete")}: ${pngFile.nativePath}`);
      
      if (isDuplicate) {
        await workingDoc.closeWithoutSaving();
      }
      return;
    }
    
    // Get image data for analysis
    updateStatus(t("processing"));
    const imageObj = await imaging.getPixels({
      documentID: workingDoc.id,
      sourceBounds: {
        left: 0,
        top: 0,
        right: workingDoc.width,
        bottom: workingDoc.height
      }
    });
    
    const fullPixelData = await imageObj.imageData.getData();
    
    // Resize to 32x32 for analysis
    const resized32x32 = resizeImageTo32x32(
      fullPixelData,
      workingDoc.width,
      workingDoc.height
    );
    
    // Analyze if image is B&W
    const isBW = analyzeImageBW(resized32x32);
    
    // Clean up pixel data
    imageObj.imageData.dispose();
    
    if (isBW) {
      updateStatus(t("outputGrayscale"));
      
      await workingDoc.changeMode(constants.ChangeMode.GRAYSCALE);
      const pngFile = await parentFolder.createFile(`${userFileName}.png`, { overwrite: true });
      await workingDoc.saveAs.png(pngFile, { compression: 6 }, true);
      
      updateStatus(`${t("complete")}: ${pngFile.nativePath}`);
      
    } else {
      updateStatus(t("outputColor"));
      
      const jpgOptFile = await parentFolder.createFile(`${userFileName}.jpg`, { overwrite: true });
      await workingDoc.saveAs.jpg(jpgOptFile, { quality: 11 }, true);
      
      updateStatus(t("complete"));
    }
    
    if (isDuplicate) {
      await workingDoc.closeWithoutSaving();
    }
    
  } catch (e) {
    updateStatus(`${t("saveFailed")}: ${e.message}`);
    console.error(e);
  }
}

// Button click handler
async function handleImageOutput() {
  try {
    const { core } = require("photoshop");
    await core.executeAsModal(handleImageOutputModal, {
      commandName: "Image Output Processing"
    });
  } catch (e) {
    updateStatus(`${t("saveFailed")}: ${e.message}`);
    console.error(e);
  }
}

module.exports = { handleImageOutput };
