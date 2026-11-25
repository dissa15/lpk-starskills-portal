const express = require('express');
const router = express.Router();
const { getAllPrograms } = require('../controllers/programController');

router.get('/', getAllPrograms);

module.exports = router;