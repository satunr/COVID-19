import {StyleSheet, View, Text} from 'react-native'


function HelpMenu({navigation}){

    return(
        <View style={styles.helpStyle}>
            <Text>Add some help info here once app is more structured</Text>
        </View>);
}

const styles = StyleSheet.create({
  
    helpStyle: {
        flex: 1,
        backgroundColor: "#D3D3D3",
    },
    
});

export default HelpMenu