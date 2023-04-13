// import * as SplashScreen from 'expo-splash-screen';
// import React, { useCallback, useEffect, useState } from 'react';
// import * as SecureStore from 'expo-secure-store';
// import { Text, View } from 'react-native';
// // secure store by expo will store the username and login on the users phone, so once they open the app we cna check if they have login
// // info and either send them to the home page, or make them login/signup.
// // function to grab user login, if its stored return true else false.
// let foundValue;
// async function getValueFor(key) {
//     let result = await SecureStore.getItemAsync(key);
//     console.log(result)
//     if (result) {
//         console.log("true");
//        foundValue = true;
//     } else {
//         foundValue = false;
//     }
// }
//
// async function save(key, value) {
//     await SecureStore.setItemAsync(key, value);
// }
// // Keep the splash screen visible while we fetch resources
// //SplashScreen.preventAutoHideAsync();
//
// function SplashScreenPage({navigation}){
//
//     const [appIsReady, setAppIsReady] = useState(false);
//
//     //use effect here because the array makes it only run once on screen open and not on every rerender
//     useEffect(() => {
//         // async function prepare(){
//         //     try{
//         //         getValueFor("username").then(r => {
//         //             console.log("returned: "+r);
//         //             if(r){
//         //                 console.log("found");
//         //                 //navigation.replace("Home");
//         //             }
//         //     }) }finally {
//         //         setAppIsReady(true);
//         //         //navigation.replace("Login");
//         //     }
//         // }
//         // // if login is stored navigate to the homepage as they already are "logged in". Check once
//         // prepare();
//         getValueFor("username").then(r => {
//             console.log("returned: " + r);
//             if (r) {
//                 console.log("found");
//                 navigation.replace("Home");
//             }
//             else{
//             navigation.replace("Login");
//             }
//         })
//
//     }, [])
//
//     // const onLayoutRootView = useCallback(async ({navigation}) => {
//     //     if (appIsReady) {
//     //         // This tells the splash screen to hide immediately! If we call this after
//     //         // `setAppIsReady`, then we may see a blank screen while the app is
//     //         // loading its initial state and rendering its first pixels. So instead,
//     //         // we hide the splash screen once we know the root view has already
//     //         // performed layout.
//     //         await SplashScreen.hideAsync();
//     //         if(foundValue){
//     //             navigation.replace("Home");
//     //         }
//     //         else{
//     //             navigation.replace("Login");
//     //         }
//     //         console.log("should be showin")
//     //
//     //     }
//     // }, [appIsReady]);
//
//     // if (!appIsReady) {
//     //     console.log("in here")
//     //     return null;
//     // }
//     return (
//         <View
//             style={{ flex: 1, alignItems: 'center', justifyContent: 'center' ,backgroundColor: "#D3D3D3",}}
//             // onLayout={onLayoutRootView}>
//             >
//             <Text>SplashScreen Demo! 👋</Text>
//         </View>
//     );
// }
// export default SplashScreenPage;