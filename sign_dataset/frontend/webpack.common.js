const HtmlWebPackPlugin = require("html-webpack-plugin");
var webpack = require('webpack');
const path = require('path');
const dotenv = require('dotenv');
const  ExtractTextPlugin = require('extract-text-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin')
const htmlWebpackPlugin = new HtmlWebPackPlugin({
  template: "./src/index.html",
  filename: "./index.html"
});

/*
const dotenvdata = require('dotenv').config( {
  path: path.join(__dirname, '.env')
} );

console.log('dotenv',dotenvdata);
*/

function common(env)
{
    return {
        entry:{
          index:'./src/index.js'
        },
        output: {
          path: path.resolve(__dirname, 'dist'),
          filename: 'js/bundle.js'
        },
        //output:'./dist/main.js',
        module: {
          rules: [
           {
             test: /\.js$/,
            exclude: /node_modules/,
            use: {
              loader: "babel-loader"
            }
          }
          ,
            {
              test: /\.(jpg|png|woff|woff2|eot|ttf|svg|gif)$/,
              use: {
                loader:'file-loader',
                options:{
                  name (file) {
                    if (env === 'development') {
                      return '[path][name].[ext]'
                    }

                    return '[hash].[ext]'
                  }
                }
              }

             },
             {
             test: /\.css$/,
             use: [
               MiniCssExtractPlugin.loader,
               "css-loader",
               "postcss-loader",
               ],
           },
          {
            test: /\.scss$/,
            use: [
                MiniCssExtractPlugin.loader,
                {
                  loader: 'css-loader'
                },
                {
                  loader: 'sass-loader',
                  options: {
                    sourceMap: true,
                    // options...
                  }
                }
              ]
          }

        ]
      },
      plugins: [
      require('tailwindcss'),
      htmlWebpackPlugin,
        new webpack.HotModuleReplacementPlugin(),
        new MiniCssExtractPlugin({
          filename: "styles.css",
          chunkFilename: "styles.css"
        }),
        new webpack.DefinePlugin({
          'process.env': JSON.stringify(dotenv.config().parsed)
      })
      ]
  }

}

module.exports = common;
