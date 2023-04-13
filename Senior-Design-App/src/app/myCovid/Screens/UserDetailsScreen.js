import {StyleSheet, View, TextInput, Pressable, FlatList, Text, Modal, Alert, KeyboardAvoidingView } from 'react-native'
import * as Location from 'expo-location';
import { database } from '../firebase';
import React, {useEffect, useState} from 'react';
import { ref, update, child, push } from "firebase/database";
import * as SecureStore from 'expo-secure-store';
//import * as TaskManager from "expo-task-manager";

//background location name
// const LOCATION_TASK_NAME = 'background-location-task';
// //task manager which is required to be at highest scope. For background tasks.
// TaskManager.defineTask(LOCATION_TASK_NAME, ({ data, error }) => {
//     if (error) {
//         // Error occurred - check `error.message` for more details.
//         return;
//     }
//     if (data) {
//         const { locations } = data;
//         location = locations[0];
//         // sets latitude/longitude by parsing json
//          let latitude = locations[0].coords.latitude;
//          let longitude = locations[0].coords.longitude;
//         // initializers for storage then stores.
//         let today = new Date();
//         let date = today.getFullYear()+'-'+(today.getMonth()+1)+'-'+today.getDate();
//         let time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
//         let infectionStatus = false; // Temporary placeholder value, connect to infection logic later
//         writeUserLocation(latitude,longitude,date,time);
//         console.log("1");
//     }
// });

//function to save settings
async function save(key, value) {
    await SecureStore.setItemAsync(key, value);
}
//function to remove setting
async function remove(setting){
    await SecureStore.deleteItemAsync(setting);
}
//error handing of text input incase user gives a bad value

//globals
let intervalID =null;
let location;
let infectedGlobal = null;


function writeUserLocation(latitude, longitude) {
    let today = new Date();
    let date = today.getFullYear()+'-'+(today.getMonth()+1)+'-'+today.getDate();
    let time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds(); 
    
    // Get a key for a new Location.
    const newLocationKey = push(child(ref(database), 'thisdoesntseemtomakeanydifference_why?')).key;
    update(ref(database, 'userLocationData/' + global.loggedUID + '/' +newLocationKey), {
        latitude: latitude,
        longitude: longitude,
        infectionStatus: infectedGlobal,
        date: date,
        time: time,
    });
}

// function updateUserInfectionStatus(infectionStatus) {
//     infectedGlobal = infectionStatus;
//     let today = new Date();
//     let date = today.getFullYear()+'-'+(today.getMonth()+1)+'-'+today.getDate();
//     let time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
//     update(ref(database, 'userInfectionData/' + global.loggedUID + '/'), {
//         infectionStatus: infectedGlobal,
//         date: date,
//         time: time,
//     });
// }

//stops grabbing location
function stopLocation(){
    if(intervalID==null) return;
    else {
        clearInterval(intervalID);
        console.log("stopped tracking location");
        remove("tracking");
    }
    //call this on user logout, but unregisteralltasks
    //TaskManager.unregisterTaskAsync(LOCATION_TASK_NAME).then(r => console.log("stopped tracking location"));
}

function UserDetailsScreen({ navigation }){
  //-----------------------------------------------------------------------------------------------------
  // list logic
  // list of locations and state to render also counter for ID.
  let initialElements = [];
  const [listState, setListState] = useState(initialElements);
  let c = 0;
  // function to update list with new elements
  const addElement = () => {
      let today = new Date();
      let date = today.getFullYear()+'-'+(today.getMonth()+1)+'-'+today.getDate();
      let time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
      let infectionStatus = infectedGlobal;
      let newArray = [...initialElements , {id : c, text: `location ${c}: Date: ${date} Time: ${time} Latitude: ${location.coords.latitude} Longitude: ${location.coords.longitude} Infected: ${infectionStatus}\n`}];
      c++;
      initialElements = newArray;
      setListState(newArray);
      console.log("added");
  }

  // moved foreground permissions here to call add element after we get location.
  const requestforegroundPermissions = async (time) => {

      // creates interval function to request location after "time" add sending data to database here.
      intervalID = setInterval(async function() {
          // request permission to track location in foreground, if status == granted grab location and print it.
          const {status} = await Location.requestForegroundPermissionsAsync();
          if (status === 'granted') {
              console.log("ran");
              location = await Location.getCurrentPositionAsync({}); //
              console.log(`Latitude: ${location.coords.latitude} Longitude: ${location.coords.longitude}`)
              addElement();
              //commenting adding to DB part out for now until new additions
              writeUserLocation(location.coords.latitude,location.coords.longitude);
          }
      }, time);
  };

  //background location tracker
  //   const requestBackgroundLocation = async (time) => {
  //       // requests foreground permissions first as that is required
  //       const {granted: fgGranted} =
  //           await Location.requestForegroundPermissionsAsync();
  //       if (!fgGranted) {
  //           return Alert.alert("Required", "Please grant GPS Location");
  //       }
  //       // then asks for background permission, (note u can not track bg location on ios expo, must use android emulator or real iphone app.)
  //       const {granted: bgGranted} =
  //           await Location.requestBackgroundPermissionsAsync();
  //
  //       if (!bgGranted) {
  //           return Alert.alert(
  //               "Location Access Required",
  //               "App requires location even when the App is backgrounded."
  //           );
  //       }
  //       // start the background task with the options set.
  //       await Location.startLocationUpdatesAsync(LOCATION_TASK_NAME, {
  //           accuracy: Location.Accuracy.BestForNavigation,
  //           timeInterval: time,
  //           foregroundService: {
  //               notificationTitle: "App Name",
  //               notificationBody: "Location is used when App is in background",
  //           },
  //           activityType: Location.ActivityType.Fitness,
  //           showsBackgroundLocationIndicator: true,
  //       });
  //   }
  //-----------------------------------------------------------------------------------------------------
  //infection tracking logic
  //Initiate modal state
    const [modalVisible, setModalVisible] = useState(false);

//-----------------------------------------------------------------------------------------------------
    const [trackingEnabled, setTrackingEnabled] = useState(false);
//changes tracking switch, also starts tracking once enabled with default value of 1 min. cancels tracking if switch is turned off.
    const toggleTracking = () => {
        setTrackingEnabled(previousState => !previousState);
        if(trackingEnabled === true){
            stopLocation();
        }
        else{
            save("tracking",trackInterval+"").then(()=>{
                requestforegroundPermissions(trackInterval);
                //requestBackgroundLocation(trackInterval);
            });
        }
    };
//-----------------------------------------------------------------------------------------------------
//new dynamic input for time.
  const [trackInterval, setTrackInterval] = useState(60000);
  const [input, setInput] = useState("");
  function changeTrackInterval(interval){
      console.log(interval);
      // if no vis change must check if tracking is enabled first
      setTrackInterval(interval * 60000);
       stopLocation();
       requestforegroundPermissions(trackInterval);
      //requestBackgroundLocation(trackInterval);
  }

return(
    <View style={styles.container}>

        <View style={styles.listContainer}>
            <FlatList
                keyExtractor={item => item.id}
                data={listState}
                renderItem={item => (<Text>{item.item.text}</Text>)}
            />
        </View>
        {/* view to bring screen up when user clicks text input*/}
      <KeyboardAvoidingView style={styles.detailsContainer} behavior={"padding"}>
          <Text style={styles.sectionText}>Toggle location tracking</Text>

          <View>
              {/*Ternary operator to show tracking settings view if tracking is enabled, otherwise tell user to enable tracking*/}
              {trackingEnabled ? <View>
                  <Text style={styles.sectionRegularText}>Location Tracked Every: {trackInterval / 60000} Minute(s)</Text>
                  <Text style={styles.sectionRegularText}>Enter Time to Track In Minutes:</Text>

                  <TextInput style={styles.sectionTextInput}
                             placeholder=" EX: 1"
                             value={input}
                             onChangeText={text => setInput(text)}>
                  </TextInput>

                  <View style={styles.detailsButtonView}>

                      <Pressable style={styles.detailsChangeButtonStyling}
                                 onPress={() => changeTrackInterval(input)}>
                          <Text style={styles.text}>Change</Text>
                      </Pressable>

                      <Pressable style={styles.locationOnButton}
                                 onPress={() => toggleTracking()}>
                          <Text style={styles.text}>Disable</Text>
                      </Pressable>

                  </View>

              </View> : <View>
                          <Text style={styles.sectionRegularText}> Enable tracking to change tracking settings!</Text>
                          <Pressable style={styles.locationOffButton} onPress={() => toggleTracking()}>
                            <Text style={styles.text}>Enable</Text>
                          </Pressable>
                        </View>
              }
          </View>

          {/*Section to track infection status*/}
            <View style={styles.infectionSection}>
                <Text style={styles.sectionText}>Track Infection Status</Text>
                {/*Pressable to track infection status*/}
                <Pressable style={styles.buttonStyling} onPress={() => setModalVisible(!modalVisible)}>
                    <Text style={styles.text}>Track</Text>
                </Pressable>
            </View>

                  {/*Container for the modal information.*/}
                      <Modal
                          animationType="slide"
                          transparent={true}
                          visible={modalVisible}
                          onRequestClose={() => {
                              Alert.alert("Modal has been closed.");
                              setModalVisible(!modalVisible);
                          }}
                      >
                          {/*the actual modal view itself that renders when button clicked*/}
                          <View style={styles.centeredView}>
                              <View style={styles.modalView}>
                                  <Text style={styles.modalText}> Confirm Infection Status</Text>

                                  <View style={styles.modalButtonView}>
                                      <Pressable style={styles.modalButtonStyling} onPress={() => { setModalVisible(!modalVisible); infectedGlobal = 1; /*updateUserInfectionStatus(1);*/ }}>
                                          <Text style={styles.text}>I am infected</Text>
                                      </Pressable>
                                      <Pressable style={styles.modalButtonStyling} onPress={() => { setModalVisible(!modalVisible); infectedGlobal = 0; /*updateUserInfectionStatus(0);*/ }}>
                                          <Text style={styles.text}>I am not infected</Text>
                                      </Pressable>
                                      <Pressable style={styles.modalButtonStyling} onPress={() => setModalVisible(!modalVisible)}>
                                          <Text style={styles.text}>Cancel</Text>
                                      </Pressable>
                                  </View>

                              </View>

                          </View>

                      </Modal>
      </KeyboardAvoidingView>
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
  // Styling for buttons, probably should use for all buttons aka pressables.
    buttonStyling: {
        alignItems: 'center',
        justifyContent: 'center',
        width: 80,
        height: 40,
        paddingHorizontal: 15,
        marginTop: 10,
        borderRadius: 10,
        marginLeft: 120,
        elevation: 3,
        backgroundColor: 'black',
    },
    modalButtonStyling: {
        alignItems: 'center',
        justifyContent: 'center',
        width: 80,
        height: 40,
        borderRadius: 10,
        elevation: 3,
        backgroundColor: 'black',
        marginLeft: 10,
    },
    listContainer: {
        flex: 4,
        width: '100%',
        borderWidth: 1,
        borderRadius: 1,
        borderStyle: "solid",
        borderColor: "black",
    },
    // Styling for text on buttons, can use for everything or not
    text: {
        fontSize: 10,
        lineHeight: 12,
        fontWeight: 'bold',
        letterSpacing: 0.15,
        color: 'white',
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
        elevation: 5,
    },
    modalText: {
        marginBottom: 20,
        fontWeight: 'bold',
        textAlign: "center",
        color: 'black',
        fontSize: 25,
    },
    detailsContainer:{
        flex: 4,
        marginTop: 5,
    },
    centeredView: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
        marginTop: 22
    },
    modalButtonView: {
        flexDirection: "row",
        alignItems: "center",
    },
    sectionText:{
        textAlign: "center",
        fontWeight: "bold",
        fontSize: 25,
        textTransform: "capitalize",
        lineHeight: 50,
    },
    sectionRegularText:{
        fontSize:16,
        letterSpacing: .15,
        lineHeight: 23,
    },
    infectionSection: {
        flex: 1,
        alignContent: "center",
    },
    switchStyling: {
        marginLeft: 100,
        marginTop: 10,
    },
    locationOnButton: {
        alignItems: 'center',
        justifyContent: 'center',
        width: 80,
        height: 40,
        paddingHorizontal: 15,
        marginTop: 10,
        borderRadius: 10,
        marginRight: 50,
        elevation: 3,
        backgroundColor: 'red',
    },
    locationOffButton: {
        alignItems: 'center',
        justifyContent: 'center',
        width: 80,
        height: 40,
        paddingHorizontal: 15,
        marginTop: 10,
        borderRadius: 10,
        marginLeft: 120,
        elevation: 3,
        backgroundColor: 'green',
    },
    sectionTextInput: {
        fontSize:16,
        letterSpacing: .15,
        lineHeight: 23,
        borderStyle: "solid",
        borderRadius: 1,
        borderColor: "black",
        borderWidth: 1.5,
        marginTop: 10
    },
    detailsButtonView: {
        flexDirection: "row",
        justifyContent: "space-between",
    },
    detailsChangeButtonStyling: {
        alignItems: 'center',
        justifyContent: 'center',
        width: 80,
        height: 40,
        paddingHorizontal: 15,
        marginTop: 10,
        borderRadius: 10,
        marginLeft: 50,
        elevation: 3,
        backgroundColor: 'black',
    },
});

export default UserDetailsScreen