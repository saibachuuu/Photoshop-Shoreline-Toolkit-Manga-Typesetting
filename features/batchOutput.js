/* batchOutput.js - Batch output feature */

const { app, action, imaging } = require("photoshop");
const uxp = require("uxp");
const { t, updateStatus } = require("../i18n/i18n");

// Import shared utilities
const { resizeImageTo32x32, analyzeImageBW } = require("./imageOutput");

// Get list of image files from folder
async function getImageFilesFromFolder(folder) {
  try {
    const entries = await folder.getEntries();
    console.log(`Found ${entries.length} total entries in folder`);
    
    const imageExtensions = [".psd", ".psb", ".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif"];
    
    const imageFiles = entries.filter(entry => {
      if (entry.isFile) {
        const lowerName = entry.name.toLowerCase();
        const isImage = imageExtensions.some(ext => lowerName.endsWith(ext));
        if (isImage) {
          console.log(`✓ Found image file: ${entry.name}`);
        }
        return isImage;
      }
      return false;
    });
    
    console.log(`Filtered to ${imageFiles.length} image files`);
    return imageFiles;
  } catch (e) {
    console.error("Error in getImageFilesFromFolder:", e);
    updateStatus(`${t("saveFailed")}: Error reading folder - ${e.message}`);
    return [];
  }
}

// Process single image file in batch
async function processImageFileInBatch(imageFile, outputFolder, executionContext) {
  try {
    const { constants } = require("photoshop");
    const { storage } = uxp;
    
    const fileName = imageFile.name.replace(/\.[^/.]+$/, "");
    
    updateStatus(`${t("openingDocument")}: ${imageFile.name}`);
    let fileToken;
    try {
      fileToken = storage.localFileSystem.createSessionToken(imageFile);
      console.log(`Created session token for: ${imageFile.name}`);
    } catch (e) {
      console.error(`Failed to create session token for ${imageFile.name}:`, e);
      return false;
    }
    
    try {
      await action.batchPlay([{
        "_obj": "open",
        "null": {
          "_path": fileToken,
          "_kind": "local"
        },
        "_isCommand": true
      }], {});
    } catch (e) {
      console.error(`Failed to open ${imageFile.name}:`, e);
      return false;
    }
    
    const doc = app.activeDocument;
    
    if (!doc) {
      console.error(`No active document after opening ${imageFile.name}`);
      return false;
    }
    
    const layerCount = doc.layers.length;
    const needsFlattening = layerCount > 1;
    
    let workingDoc;
    let isDuplicate = false;
    
    if (needsFlattening) {
      updateStatus(t("duplicating"));
      workingDoc = await doc.duplicate(`${fileName}_temp`);
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
    
    if (workingDoc.mode === constants.DocumentMode.INDEXEDCOLOR) {
      updateStatus(t("convertingToRGB"));
      await workingDoc.changeMode(constants.ChangeMode.RGB);
    }
    
    if (workingDoc.mode === constants.DocumentMode.GRAYSCALE) {
      updateStatus(t("outputGrayscale"));
      
      const pngFile = await outputFolder.createFile(`${fileName}.png`, { overwrite: true });
      await workingDoc.saveAs.png(pngFile, { compression: 6 }, true);
      
      if (isDuplicate) {
        await workingDoc.closeWithoutSaving();
      }
      
      if (!isDuplicate) {
        await doc.closeWithoutSaving();
      }
      
      return true;
    }
    
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
    
    const resized32x32 = resizeImageTo32x32(
      fullPixelData,
      workingDoc.width,
      workingDoc.height
    );
    
    const isBW = analyzeImageBW(resized32x32);
    
    imageObj.imageData.dispose();
    
    if (isBW) {
      updateStatus(t("outputGrayscale"));
      
      await workingDoc.changeMode(constants.ChangeMode.GRAYSCALE);
      const pngFile = await outputFolder.createFile(`${fileName}.png`, { overwrite: true });
      await workingDoc.saveAs.png(pngFile, { compression: 6 }, true);
      
    } else {
      updateStatus(t("outputColor"));
      
      const jpgOptFile = await outputFolder.createFile(`${fileName}.jpg`, { overwrite: true });
      await workingDoc.saveAs.jpg(jpgOptFile, { quality: 11 }, true);
    }
    
    if (isDuplicate) {
      await workingDoc.closeWithoutSaving();
    }
    
    if (!isDuplicate) {
      await doc.closeWithoutSaving();
    }
    
    return true;
    
  } catch (e) {
    console.error(`Failed to process ${imageFile.name}:`, e);
    return false;
  }
}

// Main batch output handler
async function handleBatchOutputModal(executionContext) {
  try {
    const { storage } = uxp;
    
    updateStatus(t("selectFolder"));
    const sourceFolder = await storage.localFileSystem.getFolder();
    
    if (!sourceFolder) {
      updateStatus(t("saveCancelled"));
      return;
    }
    
    console.log(`Source folder selected: ${sourceFolder.nativePath}`);
    
    updateStatus(t("creatingOutputFolder"));
    let outputFolder;
    try {
      outputFolder = await sourceFolder.getEntry("shoreline-output");
    } catch (e) {
      outputFolder = null;
    }
    
    if (!outputFolder) {
      try {
        outputFolder = await sourceFolder.createFolder("shoreline-output");
        console.log("Created output folder");
      } catch (createError) {
        updateStatus(`${t("saveFailed")}: Could not create output folder - ${createError.message}`);
        console.error("Create folder error:", createError);
        return;
      }
    } else {
      console.log("Output folder already exists");
    }
    
    const imageFiles = await getImageFilesFromFolder(sourceFolder);
    
    if (imageFiles.length === 0) {
      updateStatus("No image files found in selected folder");
      console.warn("No image files found");
      return;
    }
    
    console.log(`Processing ${imageFiles.length} files...`);
    
    let successCount = 0;
    
    for (let i = 0; i < imageFiles.length; i++) {
      const imageFile = imageFiles[i];
      console.log(`Processing file ${i + 1}/${imageFiles.length}: ${imageFile.name}`);
      
      updateStatus(`${t("batchProcessing")}: ${i + 1} ${t("of")} ${imageFiles.length} - ${imageFile.name}`);
      
      try {
        const success = await processImageFileInBatch(imageFile, outputFolder, executionContext);
        if (success) {
          successCount++;
        }
      } catch (e) {
        console.error(`Error processing ${imageFile.name}:`, e);
      }
    }
    
    updateStatus(`${t("complete")}: ${successCount} ${t("filesProcessed")}`);
    console.log(`Batch complete: ${successCount}/${imageFiles.length} files processed`);
    
  } catch (e) {
    updateStatus(`${t("saveFailed")}: ${e.message}`);
    console.error("Batch output error:", e);
  }
}

// Button click handler for batch output
async function handleBatchOutput() {
  try {
    const { core } = require("photoshop");
    await core.executeAsModal(handleBatchOutputModal, {
      commandName: "Batch Image Output Processing"
    });
  } catch (e) {
    updateStatus(`${t("saveFailed")}: ${e.message}`);
    console.error(e);
  }
}

module.exports = { 
  handleBatchOutput,
  resizeImageTo32x32,
  analyzeImageBW
};
