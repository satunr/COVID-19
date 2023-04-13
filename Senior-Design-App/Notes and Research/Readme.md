# Notes and Research Folder

- Description of any other tools, technologies and APIs needed.  
- Links to reference guides, or examples.
- Description of development environment or link to development environment
- APP:
- Uses npm, react-native, and expo for app portion. See react native docs for help.
- Utilizes the expo go app for testing:
  - to test run npx expo start and connect to server via app.
- for internal testing we can test through expo go app. https://docs.expo.dev/guides/sharing-preview-releases/#expo-go
- eas update:configure to set up the app
- eas update to push changes to expo 
- have them create a expo go account, add them to the vcu team, and have them switch to vcu account and open project.
- to run server use npx expo start (if it is stuck on connecting use --tunnel option)
- Uses a firebase realtime database to store information about user.
- User information is stored locally with encryption using https://docs.expo.dev/versions/latest/sdk/securestore/#securestoredeleteitemasynckey-options
