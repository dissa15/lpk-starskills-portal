require('dotenv').config();
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Memanggil Routes
const programRoutes = require('./src/routes/programRoutes');
app.use('/api/v1/programs', programRoutes);

// Endpoint Uji Coba Server
app.get('/api/v1/status', (req, res) => {
  res.json({ status: "success", message: "API menyala dengan sempurna!" });
});

app.listen(PORT, () => {
  console.log(`Server Back-End berjalan di http://localhost:${PORT}`);
});