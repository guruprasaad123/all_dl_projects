import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import React , { Component , Fragment, useState } from 'react';
import axios from 'axios';


const API_URL = process.env.backend || 'https://signs-app-lambda.herokuapp.com';
// const API_URL = 'http://localhost:5000';

class HeaderComponent extends Component

{
    constructor(props)
    {
        super(props);
        this.state={ navbar : false };
    }

    componentDidMount() {

      axios.get(`${API_URL}/test`).then(
        (response)=>{
         // console.log('testing',response.data)
      }).catch((error)=>{
        // console.log('error',error);
      })
    }


    upload = () => {

      // const formData = this.state.formData;

      this.setState({
        loading : true
      });

      const body = {
        'image' : this.state.preview
      }

      axios.post( `${API_URL}/upload`, body )
      .then(res => {
       // console.log('response',res);
        const data = res.data;

        if ( data.success == true) {

          this.setState({
            loading : false,
            uploading: false,
            data : data.response
          });

        }

      } ).catch((error)=>{
        this.setState({
          loading : false
        });
      })

    }

    onChange =( e )=> {

      const files = Array.from(e.target.files);

      const file = e.target.files[0];

      const reader = new FileReader();

      reader.addEventListener("load",  () => {
        // convert image file to base64 string
        const preview = reader.result;

        // console.log('preview',preview);

        this.setState({
          preview : preview
        });

      }, false);

      if (file) {
        reader.readAsDataURL(file);
      }

      this.setState({ uploading: true });

      const formData = new FormData()

      files.forEach((file, i) => {
        formData.append('file', file)
      });

      this.setState({
        formData : formData
      });


      // .then(images => {
      //   this.setState({
      //     uploading: false,
      //     images
      //   })
      // })
    }




    render()
    {

      const { data , preview , loading } = this.state;

      return (
          <Fragment>
            <div id="working" className="md:container mx-auto">

          <div className="py-12 bg-white">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="lg:text-center">
              {/* <h1 className="text-base text-indigo-600 font-semibold tracking-wide uppercase">
              Signs App
              </h1> */}
              <p className="mt-2 text-2xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
               A Basic Web App that uses AI to recognise Hand Signs (0-5)
              </p>

            </div>

            {/* <div className="mt-10">
              <dl className="space-y-10 md:space-y-0 md:grid md:grid-cols-2 md:gap-x-8 md:gap-y-10">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white">

                      <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                      </svg>
                    </div>
                  </div>
                  <div className="ml-4">
                    <dt className="text-lg leading-6 font-medium text-gray-900">
                      Competitive exchange rates
                    </dt>
                    <dd className="mt-2 text-base text-gray-500">
                      Lorem ipsum, dolor sit amet consectetur adipisicing elit. Maiores impedit perferendis suscipit eaque, iste dolor cupiditate blanditiis ratione.
                    </dd>
                  </div>
                </div>

                <div className="flex">
                  <div className="flex-shrink-0">
                    <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white">

                      <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
                      </svg>
                    </div>
                  </div>
                  <div className="ml-4">
                    <dt className="text-lg leading-6 font-medium text-gray-900">
                      No hidden fees
                    </dt>
                    <dd className="mt-2 text-base text-gray-500">
                      Lorem ipsum, dolor sit amet consectetur adipisicing elit. Maiores impedit perferendis suscipit eaque, iste dolor cupiditate blanditiis ratione.
                    </dd>
                  </div>
                </div>

                <div className="flex">
                  <div className="flex-shrink-0">
                    <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white">

                      <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </div>
                  </div>
                  <div className="ml-4">
                    <dt className="text-lg leading-6 font-medium text-gray-900">
                      Transfers are instant
                    </dt>
                    <dd className="mt-2 text-base text-gray-500">
                      Lorem ipsum, dolor sit amet consectetur adipisicing elit. Maiores impedit perferendis suscipit eaque, iste dolor cupiditate blanditiis ratione.
                    </dd>
                  </div>
                </div>

                <div className="flex">
                  <div className="flex-shrink-0">
                    <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white">

                      <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                      </svg>
                    </div>
                  </div>
                  <div className="ml-4">
                    <dt className="text-lg leading-6 font-medium text-gray-900">
                      Mobile notifications
                    </dt>
                    <dd className="mt-2 text-base text-gray-500">
                      Lorem ipsum, dolor sit amet consectetur adipisicing elit. Maiores impedit perferendis suscipit eaque, iste dolor cupiditate blanditiis ratione.
                    </dd>
                  </div>
                </div>
              </dl>
            </div>
          </div> */}


        </div>
        </div>

          <div>

        <div className="md:grid md:grid-cols-2 md:gap-x-24 md:justify-items-stretch">


        <div className="mt-5 md:mt-0 md:col-span-1 md:col-start-1 md:justify-self-center">

          <div className="p-8">


          <h2 className="text-center text-2xl text-indigo-600 font-semibold tracking-wide uppercase"> Training </h2>
          <p className="text-center mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
            This model is trained based on these training examples

          </p>
            <svg
             className="animate-bounce mx-auto mt-4 h-12 w-12 text-gray-400"
             xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 13l-7 7-7-7m14-8l-7 7-7-7" />
            </svg>
        <img className="space-y-6 mt-4 text-center" src={require('../images/SIGNS_1.png')}></img>

        <p className="mt-4 text-center max-w-2xl text-xl text-gray-500 lg:mx-auto">
               Please upload an image and give it a try , it might be able to recognize the image with 85% accuracy
              </p>

        </div>
          </div>

    <div className="mt-5 md:mt-0 md:col-span-1 md:col-start-2 md:justify-self-center">


        <div className="shadow sm:rounded-md sm:overflow-hidden">
          <div className="px-4 py-5 bg-white space-y-6 sm:p-6">

            <div>
              <label className="block text-md font-medium text-gray-700">
                Upload Picture
              </label>
              <div className="mt-2 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                <div className="space-y-1 text-center">
                  {
                  preview ?
                  <img className="mx-auto h-64 w-64" src={preview}/> :
                  <svg
                  className="mx-auto h-12 w-12 text-gray-400"
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                  aria-hidden="true"
                  >
                    <path
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    />
                  </svg>
                  }

                  <div className="flex text-sm text-gray-600">
                    <label htmlFor="file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                      <span>Upload a file</span>
                      <input id="file-upload" onChange={this.onChange}  name="file-upload" type="file" className="sr-only"/>
                    </label>
                    <p className="pl-1">or drag and drop</p>
                  </div>
                  <p className="text-xs text-gray-500">
                    PNG, JPG up to 2MB - 3MB
                  </p>
                </div>
              </div>
            </div>
          </div>
          <div className="px-4 py-3 bg-gray-50 text-center sm:px-6">

            <button onClick={this.upload}  className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
            {
            loading === true ?
            (
              <svg
              className="animate-spin h-5 w-5 mr-3"
              xmlns="http://www.w3.org/2000/svg"
              fill="none" viewBox="0 0 24 24"
              stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            ):(<noscript/>)
            }


              Predict
            </button>
          </div>
          {
             ( data ) ? (  <h1 className="text-center text-indigo-600 font-semibold tracking-wide uppercase">
              {data.class}
             </h1>) : (<noscript/>)
          }
        </div>

    </div>
        </div>

          </div>





          </div>
        </Fragment>
        );
    }

}

export default HeaderComponent;
