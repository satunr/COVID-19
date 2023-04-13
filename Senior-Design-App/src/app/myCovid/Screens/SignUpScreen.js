import {StyleSheet, View, TextInput, Button, Text} from 'react-native'
import React, { useState, useEffect } from 'react';
import { auth } from '../firebase.js';
import * as SecureStore from 'expo-secure-store';
async function save(key, value) {
    await SecureStore.setItemAsync(key, value);
}
function SignUpScreen({ navigation }){

    const [errorState, setErrorState] = useState("");

    const [email_SU, setEmail_SU] = useState('');
    const [password_SU, setPassword_SU] = useState('');
    const signupTest = () => (
      auth
        .createUserWithEmailAndPassword(email_SU, password_SU)
        .then(userCredentials => {
          const user = userCredentials.user;
          save("username",email_SU).then(()=> { // added this because on successful signup the user is logged in so we can just save and send them to home
              save("password",password_SU).then(() => {
                  save("uid",user.uid).then(() => navigation.replace("Home"))
              })
          })
          //console.log("Registered: ", user.email);
          //navigation.replace("Login");
        })
        .catch(error => {
            switch (error.code){
                case 'auth/weak-password':
                    setErrorState("The password you have selected is to weak.");
                    break;
                case 'auth/email-already-in-use':
                    setErrorState("The email is already in use! Try a different email or log in to the account.");
                    break;
                case 'auth/invalid-email':
                    setErrorState("The email is invalid. Please type in a valid email.");
                    break;
                case 'auth/operation-not-allowed':
                    setErrorState("Operation not allowed! If you see this reach out to us.");
                    break;
            }
        }, )
    );
  
    return (
      <View style={styles.container}>
  
          <View>
          <TextInput
            placeholder = "Email"
            value = {email_SU}
            onChangeText = {text => setEmail_SU(text)}
            style = {styles.input}
          />
          <TextInput
            placeholder = "Password"
            value = {password_SU}
            onChangeText = {text => setPassword_SU(text)}
            style = {styles.input}
          />
          </View>
  
          <View>
          <Button
          onPress={signupTest}
          title="Test Sign Up"
          color="#841584"
          style = {styles.buttonStyling}
          />
          </View>
          <View style={styles.errorview}>
              <Text>{errorState}</Text>
          </View>
  
      </View>
  );

  }

const styles = StyleSheet.create({
    //Styling for the entire view that holds all children on home page
    container: {
        flex: 1,
        backgroundColor: "#D3D3D3",
        alignItems: 'center',
        justifyContent: 'center',
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
});

export default SignUpScreen