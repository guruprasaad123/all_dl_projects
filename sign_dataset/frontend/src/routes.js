import React from 'react';
import { BrowserRouter as Router, Route, Switch ,HashRouter} from 'react-router-dom';
import { hashHistory ,MemoryRouter }from 'react-router';
import ResponsiveContainer from './ResponsiveContainer';
import CourseDetailsView from './components/CourseDetailsView';
import App from './App';
import TopMenuBar from './components/TopMenubar';
import FooterView from './components/FooterView';



const Root = <MemoryRouter>
              <div>            
              <TopMenuBar/>
              <Switch> 
              
                     <Route exact path="/" component={()=>ResponsiveContainer(<App/>) } />
                     <Route exact  path="/:name" component={ ({staticContext, ...props })=>ResponsiveContainer(<CourseDetailsView {...props} />) } />
                     <Route component={()=>ResponsiveContainer(<App/>)} />
             
                     </Switch>
                    <FooterView/>
                     </div>
  
              </MemoryRouter>;

export default Root;