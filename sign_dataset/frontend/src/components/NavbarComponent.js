import React , { Component } from 'react';

class NavbarComponent extends Component {

    constructor(props)
    {
        super(props);
        this.state = {
            isOpen : false
        };
    }

    toggleNavbar = ()=> {

        const { isOpen } = this.state;

        this.setState({
            isOpen : ! isOpen
        },()=>{
           // console.log('state',this.state);
        })

        }

    render() {
        const {isOpen} = this.state;
        
        return (

            <nav className="py-1 text-black bg-white-900 md:py-2 md:flex md:items-center">
            <div className="flex items-center justify-between p-2">
              {/* <!-- Menu button --> */}
              <button
              onClick={this.toggleNavbar}
              className="p-1 rounded-md md:hidden focus:outline-none focus:ring">
                <svg className="w-8 h-8" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 
                 { isOpen === false ? 
                 (           <path 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16" />
                ) : 
                 (
                    <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M6 18L18 6M6 6l12 12"
                    />) 
                 }
       


                </svg>
              </button>
              {/* <!-- Logo --> */}
              <a
                href="#"
                className="inline-block text-xl xs:text-center font-semibold tracking-wider text-indigo-600 uppercase md:ml-3"
              >
                SIGNS APP
              </a>
        
            </div>
            <div
              className={ isOpen === false ? "overflow-hidden max-h-0 md:max-h-full md:flex md:items-center md:overflow-visible md:flex-1" 
              :
              "md:max-h-full md:flex md:items-center md:overflow-visible md:flex-1" }
            >
           
                
      
              {/* <!-- Links --> */}
              <ul
                className="mx-3 mt-4 border-t border-gray-700 divide-y divide-gray-700 md:mx-1 md:flex md:items-center md:space-x-1 md:border-none md:divide-y-0 md:mt-0"
              >

                <li className="md:hidden">
                  <a
                    href="#working"
                    className="inline-flex py-2 font-medium transition-colors md:text-sm whitespace-nowrap md:p-1 md:rounded-md hover:text-gray-400 focus:text-gray-400 focus:outline-none md:focus:ring"
                    >
                    Working
                    </a>
                </li>

                <li className="md:hidden">
                  <a
                    href="#thanks"
                    className="inline-flex py-2 font-medium transition-colors md:text-sm whitespace-nowrap md:p-1 md:rounded-md hover:text-gray-400 focus:text-gray-400 focus:outline-none md:focus:ring"
                    >
                    Thanks
                    </a>
                </li>

 
              </ul>
              <ul className="flex-shrink-0 mx-3 md:ml-auto md:px-3 md:flex md:items-center md:space-x-2">

              <li className="hidden md:block">
                  <a
                    href="#working"
                    className="inline-flex py-2 font-medium transition-colors md:text-base whitespace-nowrap md:p-2 md:rounded-md hover:text-gray-400 focus:text-gray-400 focus:outline-none md:focus:ring"
                    >
                    Working
                    </a>
                </li>

                <li className="hidden md:block">
                  <a
                    href="#thanks"
                    className="inline-flex py-2 font-medium transition-colors md:text-base whitespace-nowrap md:p-2 md:rounded-md hover:text-gray-400 focus:text-gray-400 focus:outline-none md:focus:ring"
                    >
                    Thanks
                    </a>
                </li>
                
         
 
              </ul>
            </div>
          </nav>
        );
    }

}


export default NavbarComponent;