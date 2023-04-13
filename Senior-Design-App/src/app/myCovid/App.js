import { StatusBar } from 'expo-status-bar';
import {StyleSheet, Text, View, Pressable,} from 'react-native';
import {NavigationContainer} from "@react-navigation/native";
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import LoginScreen from './Screens/LoginScreen'
import HelpMenu from './Screens/HelpMenu'
import SignUpScreen from './Screens/SignUpScreen'
import UserDetailsScreen from './Screens/UserDetailsScreen';
import * as SecureStore from 'expo-secure-store';
import SplashScreenPage from "./Screens/SplashScreen";
import * as TaskManager from "expo-task-manager";
import {useState} from "react";

let stack = createNativeStackNavigator();

// TODO LIST
/*
* Implement user creation and location data associated with each user in database
* Have the details page be saved with user so if they close the page/reopen the details will be saved
* implement background tracking and have it run on all screens
* track infection status aswell and have that associated with user.
* */
//function to logout aka remove the username and password storage.
async function logout(){
    await SecureStore.deleteItemAsync("username");
    await SecureStore.deleteItemAsync("password");
    //await SecureStore.deleteItemAsync("uid"); // i dont think this is needed anymore but tbd
    TaskManager.unregisterAllTasksAsync();
}
// homescreen with navigation to details page.
function HomeScreen({ navigation }){
  return (
      //view for entire home screen
      <View style={styles.container}>
          {/*view for the entire bar that holds buttons etc*/}
          <View style={styles.barContainer}>
              {/*Changes time/battery/wifi color*/}
              <StatusBar style="auto" />

              {/*New button version allows for styling since react native button doesnt allow it*/}
              <Pressable style={styles.buttonStyling} onPress={() => navigation.navigate('User Details')}>
                  <Text style={styles.text}>Details</Text>
              </Pressable>

              {/*<Pressable style={styles.buttonStyling} onPress={() => navigation.navigate('Login')}>*/}
              {/*    <Text style={styles.text}>Login</Text>*/}
              {/*</Pressable>*/}

              <Pressable style={styles.buttonStyling} onPress={() => navigation.navigate('Help')}>
                  <Text style={styles.text}>Help</Text>
              </Pressable>
              <Pressable style={styles.logoutButtonStyling} onPress={() => {
                  logout().then(() => {
                      navigation.replace("Login")
                  })
              }}>
                  <Text style={styles.text}>Logout</Text>
              </Pressable>
          </View>
          {/*view for vizualization when that comes*/}
          <View style={styles.visualizationContainer}>

          </View>

      </View>
  );
}

 function App() {
    //screenOptions allow us to pass similar things between screens.
  return (
      <NavigationContainer >
        <stack.Navigator screenOptions={{
            //Header styling currently the red/orange
            headerStyle: {
                backgroundColor: "#6590d6",
            },
            //changes the headers title styling
            headerTitleStyle:{
                fontSize: 30,
                fontWeight: "bold",
                //color: "black",
                left: 100,
            },
            //sets color of back button and header title to black
            headerTintColor: 'black',
            // can change the name of the back button
            // headerBackTitle: 'Back',
            // sets the title to false so only the arrow shows
            headerBackTitleVisible: false
        }}>
            {/*<stack.Screen name="Splash Screen"*/}
            {/*              component={SplashScreenPage}*/}
            {/*              options={{*/}
            {/*                  title: "",*/}
            {/*              }}*/}
            {/*/>*/}
            <stack.Screen name="Login"
                          component={LoginScreen}
                          options={{
                              title: "Login",
                          }}
            />
            <stack.Screen name="Home"
                        component={HomeScreen}
                        options={{
                            title: "myCovid",
                        }}
            />
            <stack.Screen name="User Details"
                        component={UserDetailsScreen}
                        options={{
                            title: "User Details",
                            //headerStyle: "#000000",
                        }}
            />
            <stack.Screen name="Sign Up"
                        component={SignUpScreen}
                        options={{
                            title: "Sign Up",
                        }}
            />
            <stack.Screen name="Help"
                          component={HelpMenu}
                          options={{
                              title: "Help Menu",
                          }}
            />
        </stack.Navigator>
      </NavigationContainer>
  );
}


const styles = StyleSheet.create({
    //Styling for the entire view that holds all children on home page
  container: {
    flex: 1,
    backgroundColor: '#fff',
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
    //Styling for the top bar below navigation on home page
    barContainer: {
        backgroundColor: "#5A5A5A",
        width: "100%",
        flex: 1.3,
        alignContent: "space-between",
        flexDirection: "row",
    },
    // Styling for buttons, probably should use for all buttons aka pressables.
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
    // Styling for visualization on homescreen light grey currently
    visualizationContainer: {
        flex: 14,
        backgroundColor: "#D3D3D3",
        width: "100%",
    },
    titleText: {

    },
    input: {
      backgroundColor: 'white',
      paddingHorizontal: 15,
      paddingVertical: 10,
      borderRadius: 10,
      marginTop: 5,
    },
    listContainer: {
      flex: 6,
        backgroundColor: '#fff',
        width: '100%',
        //height: 100,
        borderWidth: 1
    },
    // Styling for help page
    helpStyle: {
      flex: 1,

    },
    // Styling for text on buttons, can use for everything or not
    text: {
        fontSize: 10,
        lineHeight: 12,
        fontWeight: 'bold',
        letterSpacing: 0.15,
        color: 'white',
    },
    centeredView: {
        // flex: 1,
        // justifyContent: "center",
        // alignItems: "center",
        // marginTop: 22
    },
    modalView: {
        margin: 20,
        backgroundColor: "white",
        borderRadius: 20,
        padding: 35,
        alignItems: "center",
        shadowColor: "#000",
        shadowOffset: {
            width: 0,
            height: 2
        },
        shadowOpacity: 0.25,
        shadowRadius: 4,
        elevation: 5
    },
    modalText: {
        marginBottom: 15,
        textAlign: "center"
    },
    detailsContainer:{
      flex: 4
    },
    logoutButtonStyling: {
        alignItems: 'center',
        justifyContent: 'center',
        width: 80,
        height: 40,
        paddingHorizontal: 15,
        marginTop: 10,
        marginLeft: 100,
        borderRadius: 10,
        elevation: 3,
        backgroundColor: 'red',
    }

});

export default App;


/*
function grabUserLocation(){
    //intervalID = setInterval(function() {
        const [location, setLocation] = useState(null);
        const [errorMsg, setErrorMsg] = useState(null);
        useEffect(() => {
            (async () => {
                let { status } = await Location.requestForegroundPermissionsAsync();
                if (status !== 'granted') {
                    setErrorMsg('Permission to access location was denied');
                    return;
                }

                let location = await Location.getCurrentPositionAsync({});
                setLocation(location);
            })();
        }, []);

        let text = 'Waiting..';
        if (errorMsg) {
            text = errorMsg;
        } else if (location) {
            text = JSON.stringify(location);
        }
        console.log(location);
        console.log(`latitude= ${location.latitude}`+ ` longitude= ${location.longitude}`);
   // }, 60 * 1000);

}

//will use when we implement background location keep here for now will go in details though.
//const LOCATION_TASK_NAME = 'background-location-task';
// const requestPermissions = async () => {
//     const { status } = await Location.requestBackgroundPermissionsAsync();
//     if (status === 'granted') {
//         await Location.startLocationUpdatesAsync(LOCATION_TASK_NAME, {
//             accuracy: Location.Accuracy.Balanced,
//         });
//     }
// };

// TaskManager.defineTask(LOCATION_TASK_NAME, ({ data, error }) => {
//     if (error) {
//         // Error occurred - check `error.message` for more details.
//         return;
//     }
//     if (data) {
//         const { locations } = data;
//         // do something with the locations captured in the background
//         console.log(locations);
//     }
// });

 */