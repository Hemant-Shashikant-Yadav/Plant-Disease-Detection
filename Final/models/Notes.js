const mongoose = require("mongoose");

const NotesSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "user",
    },
    baseImage: {
        type: String,
        required: true,
    },
    description: {
        type: String,
        required: false,
    },
    tag: {
        type: String,
        default: "General",
    },
    date: {
        type: Date,
        default: Date.now,
    },
    maskedImage: {
        type: String,
        required: false,
    },
});

module.exports = mongoose.model("notes", NotesSchema);
