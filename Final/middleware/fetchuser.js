const jwt = require("jsonwebtoken");
const JWT_SECRET = "ThisIsSecretKeyForToken";

const fetchuser = (req, res, nextFunction) => {
    const token = req.header("auth-token");
    if (!token) {
        res.status(401).send("Unauthorized access");
    }
    try {
        const data = jwt.verify(token, JWT_SECRET);
        req.user = data.user;
        nextFunction();
    } catch (err) {
        res.status(401).send("Unauthorized access");
    }
};

module.exports = fetchuser;
