import React , { Fragment , Component } from "react"
import styled from 'styled-components';

import CssBaseline from '@material-ui/core/CssBaseline';
import Typography from '@material-ui/core/Typography';
import Container from '@material-ui/core/Container';
import Chip from '@material-ui/core/Chip';
import PropTypes from 'prop-types';
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardHeader from '@material-ui/core/CardHeader';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import Avatar from '@material-ui/core/Avatar';
import IconButton from '@material-ui/core/IconButton';
import MoreVertIcon from '@material-ui/icons/MoreVert';
import Skeleton from '@material-ui/lab/Skeleton';
import Paper from '@material-ui/core/Paper';

import { timeAgo } from '../utils/utilities';

const useStyles = makeStyles((theme) => ({
    card: {
      maxWidth: 345,
      margin: theme.spacing(2),
    },
    media: {
      height: 190,
    },
  }));

class TweetComponent extends Component
{
    constructor(props)
    {
        super(props);
        this.state={};
    }

    render()
    {

        const { 
            loading = false ,
            value = {} ,
            tweetText="No Twitter Text Available",
            user={},
            creation ,
            lang,
            sentiment={},
            topics = []
        } = this.props;
        const timeago = timeAgo(creation);
        const sentimentState = sentiment.polarity == 0 ? 'Neutral' : (sentiment.polarity >= -0.50 ? "Negative" : sentiment.polarity >=0.50 ? "Positive" : "Neutral")
        // console.log('SentimentState => ' , sentimentState , sentiment.polarity );
        let chipData = topics.length >0 ? topics : ['others']
        const colorMap = {
             Neutral : '#ff7961',
             Positive : '#00bfa5',
             Negative : '#e57373'
         };

         const chipMap = {
            Neutral : 'default',
            Positive : 'primary',
            Negative : 'secondary'
        };

        // const classes = useStyles();

       return (

        <React.Fragment>
        <Card >
            <CardHeader
                avatar={
                loading ? (
                    <Skeleton animation="wave" variant="circle" width={40} height={40} />
                ) : (
                    <Avatar
                    alt="Avatar"
                    src={user.profile_image_url_https}
                    />
                )
                }
                // action={
                // loading ? null : (
                //     <IconButton aria-label="settings">
                //     <MoreVertIcon />
                //     </IconButton>
                // )
                // }
                title={
                loading ? (
                    <Skeleton animation="wave" height={10} width="80%" style={{ marginBottom: 6 }} />
                ) : (
                    user.name
                )
                }
                subheader={loading ? <Skeleton animation="wave" height={10} width="40%" /> : timeago }
            />
            {/* {loading ? (
                <Skeleton animation="wave" variant="rect" className={classes.media} />
            ) : (
                <CardMedia
                className={classes.media}
                image="https://pi.tedcdn.com/r/talkstar-photos.s3.amazonaws.com/uploads/72bda89f-9bbf-4685-910a-2f151c4f3a8a/NicolaSturgeon_2019T-embed.jpg?w=512"
                title="Ted talk"
                />
            )} */}

            <CardContent>
                {loading ? (
                <React.Fragment>
                    <Skeleton animation="wave" height={10} style={{ marginBottom: 6 }} />
                    <Skeleton animation="wave" height={10} width="80%" />
                </React.Fragment>
                ) : (
                    <Fragment>
                <Typography variant="body2" color="textSecondary" component="p">
                    {
               tweetText
                    }
                </Typography>

            <Typography variant="body2" color="textSecondary" component="div">
                    <b>Lang : </b>
                    <Chip variant="outlined" size="small" label={lang} />
                </Typography>        

                <Typography style={{ textAlign : 'left' , fontWeight : 'bolder' , padding:'2px' , 
                // backgroundColor: colorMap[sentimentState] 
                }} variant="body2" color="textSecondary" component="div">
                    <b>Polarity : </b>
                    <Chip variant="outlined" size="small" label={sentimentState} color={chipMap[sentimentState]} />
                </Typography>       

            </Fragment>
              
                
                )}

            <Paper component="ul" style={
                {
                    display: 'flex',
                    justifyContent: 'center',
                    flexWrap: 'wrap',
                    listStyle: 'none',
                    padding: '0.5rem',
                    margin: 0,
                  }
            } >
                {chipData.map((data,index) => {
      
                    return (
                    <li key={index}>
                        <Chip
                        style={
                            {
                                margin: '0.5rem'
                            }
                        }
                        label={data}
                        />
                    </li>
                    );
                })}
                </Paper>
                
            </CardContent>
    </Card>
      </React.Fragment>

       )
    }
}

export default TweetComponent;