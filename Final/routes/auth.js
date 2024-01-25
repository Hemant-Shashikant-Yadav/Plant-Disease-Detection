const express = require("express");
const User = require("../models/User");
const router = express.Router();
const { body, validationResult } = require("express-validator");
const bcrypt = require("bcryptjs");
const fetchuser = require("../middleware/fetchuser");
const jwt = require("jsonwebtoken");

const JWT_SECRET = "ThisIsSecretKeyForToken";

const validatorArray = [
    body("name", "the value of name should be atleast 5").isLength({ min: 5 }),
    body("password", "Enter strong password").isLength({ min: 8 }),
    body("email", "Enter valid email, please").isEmail(),
];

// function to encrypt the password before storing it's hash value onto the database
const createSecurePassword = async (userPassword) => {
    const salt = await bcrypt.genSalt(10);
    let securePassword = await bcrypt.hash(userPassword, salt);
    return securePassword;
};

// endpoint -->api/auth/createuser . No login required
router.post("/createuser", validatorArray, async (request, response) => {
    //checking for validation errors
    const errors = validationResult(request);
    // if errors are there return
    if (!errors.isEmpty()) {
        return response.status(400).json({ errors: errors.array() });
    }
    try {
        //checking if user does already exist with the requested email-id
        let user = await User.findOne({ email: request.body.email });
        // if user already exist return
        if (user) {
            return response
                .status(400)
                .json({ error: "User with this email already exists" });
        }

        // if user doesn't exist then create a new user
        user = await User.create({
            name: request.body.name,
            password: await createSecurePassword(request.body.password),
            email: request.body.email,
        });
        const data = {
            user: {
                id: user.id,
            },
        };
        const username = request.body.name;
        const authToken = jwt.sign(data, JWT_SECRET);
        // user creation success
        response.json({ authToken, username });
    } catch (err) {
        console.log(err.message);
        response.status(500).send("Error contacting database");
    }
});

// endpoint --> api/auth/login Authenticate a user. No login required
router.post(
    "/login",
    [
        body("email", "enter valid email").isEmail(),
        body("password", "enter valid password").isLength({ min: 8 }),
    ],
    async (request, response) => {
        //checking for validation errors
        const errors = validationResult(request);
        // if errors are there return
        if (!errors.isEmpty()) {
            return response.status(400).json({ errors: errors.array() });
        }

        const { email, password } = request.body;
        try {
            let user = await User.findOne({ email });
            if (!user) {
                return response
                    .status(400)
                    .json({ error: "Invalid credentials" });
            }
            const comparePassword = await bcrypt.compare(
                password,
                user.password
            );
            if (!comparePassword) {
                return response
                    .status(400)
                    .json({ error: "Invalid credentials" });
            }
            const data = {
                user: {
                    id: user.id,
                },
            };
            const username = user.name;

            const authToken = jwt.sign(data, JWT_SECRET);

            // user login success
            response.json({ authToken, username });
        } catch (err) {
            response.status(500).send("Database connection failed");
        }
    }
);

// endpoint --> /api/auth/getuser. Login Required

router.post("/getuser", fetchuser, async (request, response) => {
    try {
        let userId = request.user.id;
        const user = await User.findById(userId).select("-password");
        response.send(user);
    } catch (error) {
        response.status(500).send("Database connection failed");
    }
});

module.exports = router;
