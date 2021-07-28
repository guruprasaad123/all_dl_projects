import firebase from 'firebase';

export const config = {
    apiKey: "AIzaSyAMAVFA0QwffEg5Y-AtrNwXtIefsQuqXOI",
    authDomain: "startup-c521c.firebaseapp.com",
    databaseURL: "https://startup-c521c.firebaseio.com",
    projectId: "startup-c521c",
    storageBucket: "startup-c521c.appspot.com",
    messagingSenderId: "1087133454518"
  };

  export function initFb()
  {
      
        firebase.initializeApp(config);
        console.log('Fb initialised');
  }

  export function rootRef()
  {
      return firebase.database().ref();
  }

  export function getFunction()
  {
   return firebase.functions().httpsCallable('feedbackMail');
  }
