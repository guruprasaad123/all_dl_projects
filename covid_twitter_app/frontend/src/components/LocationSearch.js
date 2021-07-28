import React , {Component , Fragment } from 'react';
import axios from 'axios';

import TextField from '@material-ui/core/TextField';
import Autocomplete from '@material-ui/lab/Autocomplete';
import CircularProgress from '@material-ui/core/CircularProgress';
import Typography from '@material-ui/core/Typography';

import IconButton from '@material-ui/core/IconButton';
import Input from '@material-ui/core/Input';
import FilledInput from '@material-ui/core/FilledInput';
import OutlinedInput from '@material-ui/core/OutlinedInput';
import InputLabel from '@material-ui/core/InputLabel';
import InputAdornment from '@material-ui/core/InputAdornment';
import FormHelperText from '@material-ui/core/FormHelperText';
import FormControl from '@material-ui/core/FormControl';

import Visibility from '@material-ui/icons/Visibility';
import VisibilityOff from '@material-ui/icons/VisibilityOff';

import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';

import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';

import Button from '@material-ui/core/Button';

import Chip from '@material-ui/core/Chip';

const Headers = {"Access-Control-Allow-Origin": "*"};
axios.defaults.headers.common['Access-Control-Allow-Origin'] = "*";

// Set config defaults when creating the instance
const axiosInstance = axios.create({
     headers : Headers
  });

const {google_key = ''} = process.env;

export class LocationSearch extends Component {
    
    constructor(props)
    {
        super(props);
        this.state= {
            places : [ { title : 'Florence, SC, USA' , disable : false }]
        };
    }

    getPlaceSuggestions = (input)=>{

        axiosInstance.get(`https://maps.googleapis.com/maps/api/place/autocomplete/json?input=${input}&ype=geocode&language=en&key=${google_key}`).then((object)=>{
        const data = object.data;
        const places = data.predictions.map( (place) => ({ title : place.description , disable : false }) );
        
        this.setState({
            places : places
        });

        }).catch((error)=>{
            console.log('Error => ',error);
            console.log('Message => ',error.message);
        });
    }

    getGeocode = (location)=>{

        return new Promise( (resolve , reject)=>{

        axiosInstance.get(`https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSyDTZa7R_9Z2_cFR8GJa2RYhmvzaPa5DjNc&address=${location}`).then((object)=>{
            const data = object.data;
            if( typeof data.results[0].geometry === 'object' )
            {
                const location = data.results[0].geometry.location ; // { lat : '' , lng : '' }

                this.setState({
                    geocode : location ,
                    loading : false
                } , ()=>{
                    resolve(location);
                });

            }else
            {
              console.log('Geocode not found ')
            }
            
            }).catch((error)=>{
                console.log('Error => ',error);
                console.log('Message => ',error.message);

                this.setState({
                    location : [''] ,
                    loading : false
                },()=>{
                    reject(error);
                });
            });
        });

    }

    onTyping = (e) =>{
        console.log('value => ',e.target.value);
        const string = e.target.value ;
        if ( string && string.length && string.length >=3 )
        {
            this.setState({
                loading : true
            });
           // this.getPlaceSuggestions(string);
        }
    }

    toggleGeoSearch = (e) =>{
        console.log('value => ',e.target.name);
        const name = e.target.name ;

        if( name === 'geoSearch') 
        {
            const toggle = ! Boolean(this.state.geoSearch) ;

            this.setState({
                geoSearch : toggle
            })
        }
    }

    applySearchParams = ()=>{
        const {location = null , radius = null } = this.state ;
        
        console.log('Applied => ',location , radius );

        this.getGeocode(location).then((geoObject)=>{

        }).catch((error)=>{
            console.log('error => ',error);
        })
    }

    handleLocation = (e)=>{
        console.log(e.target.value);
        const location = e.target.value ;
        const { places } = this.state;
        this.setState({
            location : places[location].title
        });
    }

    handleRadius = (e)=>{
        console.log(e.target.value);
        const radius = e.target.value ;

        this.setState({
            radius : radius
        });

    }

    render()
    {

        const { 
            places = [ { title : 'Florence, SC, USA' , disable : false }] ,
            open = false ,
            loading = false ,
            geoSearch = false ,
            
        } = this.state;

        return <Fragment>
                     
                    <Grid container spacing={3}>
                    <Grid item xs={12}>
                    <Paper style={{
                        textAlign: 'center'
                    }}>

                    <FormGroup style={{
                        justifyContent : 'center'
                    }} row>
                        <FormControlLabel
                            control={<Checkbox color="primary" checked={geoSearch} onChange={this.toggleGeoSearch} name="geoSearch" />}
                            label="Apply Geo Search"
                        />
                    </FormGroup>

                           <Autocomplete
                                id="combo-box-demo"
                                disabled={!geoSearch}
                                options={places}
                                getOptionLabel={(option) => option.title}
                                loading={loading}
                                onChange={this.handleLocation}
                                getOptionSelected={(option, value) => {
                                    console.log('options selected : ',option , value );
                         
                                return    option.title === value.title
                                }}
                                getOptionDisabled={(option) => option.disable === true }
                                style={{ 
                                    width: 300,
                                    margin : 'auto'
                                }}
                                renderInput={(params) => <TextField 
                                                    {...params} 
                                                    label="Places" 
                                                    disabled={!geoSearch}
                                                    variant="outlined" 
                                                    onChange={this.onTyping}
                                                    InputProps={{
                                                        ...params.InputProps,
                                                        endAdornment: (
                                                        <Fragment>
                                                            {loading ? <CircularProgress color="inherit" size={20} /> : null}
                                                            {params.InputProps.endAdornment}
                                                        </Fragment>
                                                        ),
                                                    }}
                                                    />}
                                />


                            <FormControl   style={{ 
                                width: 300 ,
                                margin : '1rem'
                                }} variant="outlined">
                                    <OutlinedInput
                                        id="outlined-adornment-weight"
                                        // value={values.weight}
                                        // onChange={handleChange('weight')}
                                        disabled={!geoSearch}
                                        onChange={this.handleRadius}
                                        endAdornment={<InputAdornment position="end">miles</InputAdornment>}
                                        aria-describedby="outlined-weight-helper-text"
                                        inputProps={{
                                        'aria-label': 'weight',
                                        'type': 'number'
                                        }}
                                        labelWidth={0}
                                    />
                                    <FormHelperText id="outlined-weight-helper-text">Radius</FormHelperText>
                                    </FormControl>

                                <FormGroup 
                                style={{
                                justifyContent : 'center'
                                }} 
                                row
                                >
                                <Button 
                                disabled={!geoSearch}
                                onClick={this.applySearchParams}
                                variant="contained" 
                                color="primary">
                                    Apply
                                </Button>
                    </FormGroup>

                    <Typography style={{ 
                        margin : '1rem',
                        textAlign : 'center' ,
                        fontWeight : 'bolder' , 
                        padding:'2px' , 
                        }} 
                        variant="h2" 
                        color="textSecondary" 
                        component="h2"
                        >
                        <b>Search : </b>
                        <Chip variant="outlined" size="small" label={'worldwide'} color={'secondary'} />
                </Typography>       
            
                        </Paper>
                    </Grid>
                
                 </Grid>
        
        </Fragment>  ;
    }

}

export default LocationSearch;
