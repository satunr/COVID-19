// Import the functions you need from the SDKs you need
import * as firebase from 'firebase/compat';
import {getDatabase} from "firebase/database";
//import 'firebase/compat/auth';
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBhPX8Mq-2go0aKUo7V59mxIsuTLnRkd1A",
  authDomain: "capstone-fa57b.firebaseapp.com",
  projectId: "capstone-fa57b",
  storageBucket: "capstone-fa57b.appspot.com",
  messagingSenderId: "1011881277737",
  appId: "1:1011881277737:web:97749bae0bfd5295a95f38",
  measurementId: "G-4E4KX300JV",
    databaseURL: "https://capstone-fa57b-default-rtdb.firebaseio.com/",
};

// Initialize Firebase
let app;
if (firebase.apps.length === 0)
{
    app = firebase.initializeApp(firebaseConfig);
}
else
{
    app = firebase.app();
}

const auth = firebase.auth();
const database = getDatabase(app);
export { auth,database };
