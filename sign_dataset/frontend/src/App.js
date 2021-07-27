import React , { Component, Fragment } from 'react';
import styled from 'styled-components';
import FooterComponent from './components/FooterComponent';
import HeaderComponent from './components/HeaderComponent';
import NavbarComponent from './components/NavbarComponent';
import ThanksComponent from './components/ThanksComponent';

class App extends Component
{
    constructor(props)
    {
        super(props);
        this.state={};
    }

    render()
    {
       return (

    <Fragment>
        <NavbarComponent/>
    <HeaderComponent/>
    <ThanksComponent/>
    <FooterComponent/>
    </Fragment>

    )
    }
}

export default App;
