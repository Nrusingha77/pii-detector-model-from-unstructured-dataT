const mongoose = require("mongoose");

const PiiRecordSchema = new mongoose.Schema({
  text: String,
  pii_detected: Array,
  timestamp: { type: Date, default: Date.now },
});

module.exports = mongoose.model("PiiRecord", PiiRecordSchema);
