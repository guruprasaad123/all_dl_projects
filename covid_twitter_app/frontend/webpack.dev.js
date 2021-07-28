const merge = require('webpack-merge');
const common = require('./webpack.common.js');
const path = require('path');
module.exports = merge( common('development'),{
  mode:'development',
  devServer: {
    historyApiFallback: {
      index: '/'
    },
    contentBase: path.join(__dirname, 'src'),
    compress: true,
    port: 8000
  }
});
