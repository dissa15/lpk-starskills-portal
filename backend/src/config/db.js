const mysql = require('mysql2/promise');

const pool = mysql.createPool({
  host: '127.0.0.1',
  port: 3306,
  user: 'root',
  password: '',
  database: 'lpk_starskills_db'
});

pool.getConnection()
  .then((conn) => {
    console.log('✅ Database MySQL Laragon LPK Star Skills berhasil terhubung!');
    conn.release();
  })
  .catch((err) => {
    console.error('❌ Koneksi Database Gagal:', err.message);
  });

module.exports = pool;