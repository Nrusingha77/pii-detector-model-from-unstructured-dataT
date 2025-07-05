const express = require("express");
const router = express.Router();
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");

const upload = multer({ storage: multer.memoryStorage() });

router.post("/process", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: "No file provided",
      });
    }

    const formData = new FormData();
    formData.append("file", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });
    formData.append("mode", req.body.mode || "mask");

    const response = await axios.post(
      `${process.env.FASTAPI_URL}/files/process-file`,
      formData,
      {
        headers: {
          ...formData.getHeaders(),
        },
      }
    );

    // Make sure to use the complete FastAPI URL
    const pdfUrl = `${process.env.FASTAPI_URL}${response.data.pdf_download_url}`;

    res.json({
      success: true,
      data: response.data,
      pdfUrl: pdfUrl,
    });
  } catch (error) {
    console.error("Error details:", error.response?.data || error.message);
    res.status(500).json({
      success: false,
      error: "Failed to process file",
      details: error.response?.data || error.message,
    });
  }
});

// Update proxy download route
router.get("/download/:filename", async (req, res) => {
  try {
    const response = await axios.get(
      `${process.env.FASTAPI_URL}/files/download/${req.params.filename}`,
      {
        responseType: "stream",
        validateStatus: false,
      }
    );

    if (response.status === 404) {
      return res.status(404).json({
        success: false,
        error: "File not found",
      });
    }

    res.setHeader("Content-Type", "application/pdf");
    res.setHeader(
      "Content-Disposition",
      `attachment; filename=${req.params.filename}`
    );
    response.data.pipe(res);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to download file",
    });
  }
});

module.exports = router;
