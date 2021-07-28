import React , { Component } from "react"
import styled from 'styled-components';

import CssBaseline from '@material-ui/core/CssBaseline';
import Typography from '@material-ui/core/Typography';
import Container from '@material-ui/core/Container';
import ListView from './components/ListView';

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

        <React.Fragment>
        <CssBaseline />
        <Container maxWidth="lg">
          <Typography   variant="h2" component="h2" style={{ textAlign : 'center' , fontWeight : 'bolder' , backgroundColor: '#cfe8fc' }} >
            Welcome to #Corona Analysis App 
              </Typography>
              <ListView/>
        </Container>

       
      
      </React.Fragment>

       )
    }
}

export default App;