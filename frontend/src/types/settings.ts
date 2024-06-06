export const Languages = {
  zh: '简体中文',
  dev: 'English' // i18n default
};
export type Language = keyof typeof Languages;
export type SettingsType = {
  language: Language
  darkMode: boolean
  autoUpdatesCheck: boolean
  giteeUpdatesSource: boolean
  cnMirror: boolean
  useHfMirror: boolean
  host: string
  dpiScaling: number
  customModelsPath: string
  customPythonPath: string
  apiUrl: string
  apiKey: string
  apiChatModelName: string
  apiCompletionModelName: string
  coreApiUrl: string
  username: string
  password: string
  email: string
  phoneNumber: string
  code: string
  token: string
  oldCode: String
  uuid: String
  privacyConfirmed: boolean
  notInteresting: boolean
  notTruth: boolean
  timeout: boolean
  notLogin: boolean
  notChat: boolean
  bodyHit: boolean
  policy: boolean
  sex: boolean
  notHealthy: boolean
  others: boolean
  description: string
  contactPhone: string
  contactEmail: string
}