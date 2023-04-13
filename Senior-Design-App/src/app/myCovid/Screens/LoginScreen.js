import {StyleSheet, View, TextInput, Pressable, Text} from 'react-native'
import React, { useState, useEffect } from 'react';
import { auth } from '../firebase.js';
import * as SecureStore from 'expo-secure-store';
// gets stored value, returns it or false
async function getValueFor(key) {
    let result = await SecureStore.getItemAsync(key);
    if (result) {
       return result;
    } else {
        return false;
    }
}
//function to save login info for user so they dont get prompt every time
async function save(key, value) {
    await SecureStore.setItemAsync(key, value);
}

if (global.loggedUID == null)
{
  global.loggedUID = "unknown_user"; // Value resets whenever ExpoGo refreshes the app after a code change!
}

function LoginScreen({ navigation }){
    const [errorState, setErrorState] = useState("");
    const [email_LI, setEmail_LI] = useState('');
    const [password_LI, setPassword_LI] = useState('');
  //use effect here because the array makes it only run once on screen open and not on every rerender
    useEffect(() => {
        //intial vars to log in with
        let username;
        let password;
        // checks if we have username stored, if yes set username then check if password is stored.
        getValueFor("username").then(r => {
            if(r){
                username = r;
                // if password is stored then sign in with firebase so we are authenticated and set uid, then navigate to the home screen.
                getValueFor("password").then((r) => {
                    if(r){
                        password = r;
                        console.log(username +" "+ password);
                        auth.signInWithEmailAndPassword(username, password).then(userCredentials => {
                            const  user = userCredentials.user;
                            global.loggedUID = user.uid;
                            navigation.replace("Home");
                        })
                    }
                })
            }
        })
    }, [])

  const loginTest = () => (
    auth
      .signInWithEmailAndPassword(email_LI, password_LI)
      .then(userCredentials => {
        const user = userCredentials.user;
        global.loggedUID = user.uid;
        console.log("Logged in: ", user.email);
          save("username", email_LI).then(() => { // saves user info for next time.
              save("password",password_LI).then(() => {
                  save("uid",user.uid).then(()=> navigation.replace('Home'));
                   // Redirect user to login after successful sign-up
              } )
          })
        //navigation.replace('Home'); // Redirect user to home after successful login, changed this as it makes sense to show the homepage and allow them to go to detials if they want.
      })
      .catch(error => {
          switch (error.code){
              case 'auth/wrong-password':
                  setErrorState("The password you have input is wrong. It is case sensitive.");
                  break;
              case 'auth/user-disabled':
                  setErrorState("The account is disabled. Please reach out for support.");
                  break;
              case 'auth/invalid-email':
                  setErrorState("The email is invalid. Please type in a valid email.");
                  break;
              case 'auth/user-not-found':
                  setErrorState("User not found! Please go to the signup page and make an account.");
                  break;
          }
      }, )
  );

  return (
    
    <View style={styles.container}>

        <View style={styles.container}>
            <TextInput
                placeholder = "Email"
                value = {email_LI}
                onChangeText = {text => setEmail_LI(text)}
                style = {styles.input}
            />

            <TextInput
                placeholder = "Password"
                value = {password_LI}
                onChangeText = {text => setPassword_LI(text)}
                style = {styles.input}
            />

            <Pressable style={styles.buttonStyling} onPress={() => loginTest()}>
                <Text style={styles.text}>Login</Text>
            </Pressable>

            <Pressable style={styles.buttonStyling} onPress={() => navigation.navigate('Sign Up')}>
                <Text style={styles.text}>Register</Text>
            </Pressable>

            <View style={styles.errorview}>
                <Text>{errorState}</Text>
            </View>

        </View>

    </View>
);
}

const styles = StyleSheet.create({
container: {
  flex: 1,
    backgroundColor: "#D3D3D3",
  alignItems: 'center',
  justifyContent: 'center',
},
row: {
  flexDirection:"row",
  backgroundColor: '#fff',
  borderColor: 'black',
    flex: 4,
  //justifyContent: 'space-between'
},
  barContainer: {
      backgroundColor: "#5A5A5A",
      width: "100%",
      flex: 1.3,
      alignContent: "space-between",
      flexDirection: "row",
  },
  buttonStyling: {
      alignItems: 'center',
      justifyContent: 'center',
      width: 80,
      height: 40,
      paddingHorizontal: 15,
      marginTop: 10,
      marginLeft: 10,
      borderRadius: 10,
      elevation: 3,
      backgroundColor: 'black',
  },
  
  input: {
    backgroundColor: 'white',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 10,
    marginTop: 5,
  },
    text: {
        fontSize: 10,
        lineHeight: 12,
        fontWeight: 'bold',
        letterSpacing: 0.15,
        color: 'white',
    },

});

export default LoginScreen