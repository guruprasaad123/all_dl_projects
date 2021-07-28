import React , {Component, Fragment} from 'react';
import axios from 'axios';
import InfiniteScroll from 'react-infinite-scroller';
import TweetComponent from './TweetComponent';
import LocationSearch from './LocationSearch';
// http://ec2-13-232-219-139.ap-south-1.compute.amazonaws.com
const { api_url =  'http://ec2-13-232-219-139.ap-south-1.compute.amazonaws.com' , port=80 } = process.env;

function useTweets()
{
    return 
}

class ListView extends Component {

    constructor(props)
    {
        super(props);
        this.state = {};
        console.log('host : ', process.env )
    }

    componentDidMount(props)
    {
        this.loadTweets();
    }


    setGeocode = ( { lat = '' , lng = '' , radius = ''} )=>{
        
        const geoSearch = `?lat=${lat}&lng=${lng}&radius=${radius}mi`;
        
        return geoSearch;
    }

    loadTweets = ()=>
    {
        const geocode = this.setGeocode( {
            "lat" : 43.7695604,
            "lng" : 11.2558136,
            radius : 1
         }
         );

        axios.get(`${api_url}:${port}/api/corona/0/9999${geocode}`).then((object)=>{
              
        const response = object.data.response ;
        console.log('response => ',  response.length ); 
        console.log('response => 1 ',JSON.parse(response.tweets))
            if( response.length > 0 )
            {
                this.setState({ tweets : JSON.parse(response.tweets) , searchMeta : response.search_metadata });
            }
            
        }).catch((error)=>{
            console.log('Error => ',error);
            console.log('Message => ',error.message);
        })
    }

    scrollTop = ()=>{
        const doc = document.getElementById('scrollBar');
        doc.scrollTop = 0;
      }
    
    getPlaceSuggestions = (input)=>{

        axios.get(`https://maps.googleapis.com/maps/api/place/autocomplete/json?input=${input}&ype=geocode&language=en&key=AIzaSyBuh3nvxDKSq3tlS0kM8x6glR55M5v_9B8`).then((object)=>{
        const data = object.data;
        const places = data.predictions.map((place)=>place.description);
        
        this.setState({
            places : places
        });

        }).catch((error)=>{
            console.log('Error => ',error);
            console.log('Message => ',error.message);
        });
    }

    getGeocode = (location)=>{

        axios.get(`https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSyDTZa7R_9Z2_cFR8GJa2RYhmvzaPa5DjNc&address=${location}`).then((object)=>{
            const data = object.data;
            if( typeof data.results[0].geometry === 'object' )
            {
                const location = data.results[0].geometry.location ; // { lat : '' , lng : '' }

                this.setState({
                    location : location
                });

            }else
            {
              console.log('Geocode not found ')
            }
            
            }).catch((error)=>{
                console.log('Error => ',error);
                console.log('Message => ',error.message);
            });
    }

    loadMore = ()=>
    {
        const searchMeta = this.state.searchMeta ;
        axios.get(`${api_url}:${port}/api/corona/1/${searchMeta.max_id}`).then((object)=>{
            const response = object.data.response ;
            console.log('response => ',response.length);
            console.log('response => 1 ',JSON.parse(response.tweets))
            if( response.length > 0 )
            {
                this.setState({ 
                    tweets : 
                        [
                        ...JSON.parse(response.tweets),
                        ...this.state.tweets 
                    ],
                    
                    searchMeta : response.search_metadata } , ()=>{
                        console.log('response => 1 ',response.tweets[0])
                        this.scrollTop();
                    });
            }
        }).catch((error)=>{
            console.log('Error => ',error);
            console.log('Message => ',error.message);
        })
    }

    render()
    {
        const tweets = this.state.tweets ;

        return (
            <Fragment >
            {
                /*
                <LocationSearch/>
                */
            }

            <div id="scrollBar" style={ { height : '700px' , overflow: 'auto'}}>
              {
            (tweets && Array.isArray(tweets) )?
             <InfiniteScroll
                    pageStart={0}
                    loadMore={this.loadMore}
                    hasMore={true || false}
                    loader={<div className="loader" key={0}>Loading ...</div>}
                    useWindow={false}
                >
                    {tweets.map( (object,index)=>{
                        return (
                            <TweetComponent 
                            key={index}
                            value={object}
                            tweetText={object.text}
                            user={object.user}
                            creation={object.created_at}
                            lang={object.lang}
                            topics={object.topics}
                            sentiment={object.sentiment}
                            >        
                            </TweetComponent>
                        )
                    })}
                </InfiniteScroll>
            : <p style={{
                textAlign:'center'
            }}> Loading Tweets please wait ... </p>
            }
               
            </div>
            
            </Fragment>
        );
    }
}

export default ListView;
