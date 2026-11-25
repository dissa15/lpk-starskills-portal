const pool = require('../config/db');

const getAllPrograms = async (req, res) => {
  try {
    const [rows] = await pool.query('SELECT * FROM program_pelatihan');
    res.status(200).json({ status: 'success', data: rows });
  } catch (error) {
    console.error(error);
    res.status(500).json({ status: 'error', message: 'Gagal mengambil data' });
  }
};

module.exports = { getAllPrograms };