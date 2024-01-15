import commonStore, { MonitorData, Platform } from './stores/commonStore';
import { FileExists, GetPlatform, ListDirFiles, ReadJson } from '../wailsjs/go/backend_golang/App';
import { Cache, checkUpdate, downloadProgramFiles, LocalConfig, refreshLocalModels, refreshModels } from './utils';
import { getStatus } from './apis';
import { EventsOn, WindowSetTitle } from '../wailsjs/runtime';
import manifest from '../../manifest.json';
import { defaultModelConfigs, defaultModelConfigsMac } from './pages/defaultConfigs';
import { t } from 'i18next';
import { Preset } from './types/presets';
import { toast } from 'react-toastify';
import { MidiMessage, MidiPort } from './types/composition';

export async function startup() {
  initPresets();

  await GetPlatform().then(p => commonStore.setPlatform(p as Platform));

  if (commonStore.platform !== 'web') {
    downloadProgramFiles();
    EventsOn('downloadList', (data) => {
      if (data)
        commonStore.setDownloadList(data);
    });
    EventsOn('wsl', (await import('./pages/Train')).wslHandler);
    EventsOn('wslerr', (e) => {
      console.log(e);
    });
    initLocalModelsNotify();
    initLoraModels();
    initHardwareMonitor();
    initMidi();
  }

  await initConfig();

  if (commonStore.platform !== 'web') {
    initCache(true).then(initRemoteText); // depends on config customModelsPath

    if (commonStore.settings.autoUpdatesCheck) // depends on config settings
      checkUpdate();

    getStatus(1000).then(status => { // depends on config api port
      if (status)
        commonStore.setStatus(status);
    });
  }
}

async function initRemoteText() {
  await fetch('https://cdn.jsdelivr.net/gh/josstorer/RWKV-Runner@master/manifest.json', { cache: 'no-cache' })
  .then(r => r.json()).then((data) => {
    if (data.version >= manifest.version) {
      if (data.introduction)
        commonStore.setIntroduction(data.introduction);
      if (data.about)
        commonStore.setAbout(data.about);
    }
  });
}

async function initConfig() {
  await ReadJson('config.json').then((configData: LocalConfig) => {
    if (configData.modelSourceManifestList)
      commonStore.setModelSourceManifestList(configData.modelSourceManifestList);

    if (configData.settings)
      commonStore.setSettings(configData.settings, false);

    if (configData.dataProcessParams)
      commonStore.setDataProcessParams(configData.dataProcessParams, false);

    if (configData.loraFinetuneParams)
      commonStore.setLoraFinetuneParameters(configData.loraFinetuneParams, false);

    if (configData.modelConfigs && Array.isArray(configData.modelConfigs))
      commonStore.setModelConfigs(configData.modelConfigs, false);
    else throw new Error('Invalid config.json');
    if (configData.currentModelConfigIndex &&
      configData.currentModelConfigIndex >= 0 && configData.currentModelConfigIndex < configData.modelConfigs.length)
      commonStore.setCurrentConfigIndex(configData.currentModelConfigIndex, false);
  }).catch(() => {
    commonStore.setModelConfigs(commonStore.platform !== 'darwin' ? defaultModelConfigs : defaultModelConfigsMac, true);
  });
}

async function initCache(initUnfinishedModels: boolean) {
  await ReadJson('cache.json').then((cacheData: Cache) => {
    if (cacheData.version === manifest.version && cacheData.depComplete)
      commonStore.setDepComplete(cacheData.depComplete, false);
  }).catch(() => {
  });
  await refreshModels(false, initUnfinishedModels);
}

async function initPresets() {
  await ReadJson('presets.json').then((presets: Preset[]) => {
    if (Array.isArray(presets))
      commonStore.setPresets(presets, false);
  }).catch(() => {
  });
}

async function initLoraModels() {
  const refreshLoraModels = () => {
    ListDirFiles('lora-models').then((data) => {
      if (!data) return;
      const loraModels = [];
      for (const f of data) {
        if (!f.isDir && f.name.endsWith('.pth')) {
          loraModels.push(f.name);
        }
      }
      commonStore.setLoraModels(loraModels);
    });
  };

  refreshLoraModels();
  EventsOn('fsnotify', (data: string) => {
    if (data.includes('lora-models'))
      refreshLoraModels();
  });
}

async function initLocalModelsNotify() {
  EventsOn('fsnotify', (data: string) => {
    if (data.includes('models') && !data.includes('lora-models'))
      refreshLocalModels({ models: commonStore.modelSourceList }, false); //TODO fix bug that only add models
  });
}

async function initHardwareMonitor() {
  EventsOn('monitor', (data: string) => {
    const results: MonitorData = JSON.parse(data);
    if (results) {
      commonStore.setMonitorData(results);
      // WindowSetTitle(`RWKV-Runner (${t('RAM')}: ${results.usedMemory.toFixed(1)}/${results.totalMemory.toFixed(1)} GB, ${t('VRAM')}: ${(results.usedVram / 1024).toFixed(1)}/${(results.totalVram / 1024).toFixed(1)} GB, ${t('GPU Usage')}: ${results.gpuUsage}%)`);
      WindowSetTitle(`Luxi-NLMChat (${t('RAM')}: ${results.usedMemory.toFixed(1)}/${results.totalMemory.toFixed(1)} GB, ${t('VRAM')}: ${(results.usedVram / 1024).toFixed(1)}/${(results.totalVram / 1024).toFixed(1)} GB, ${t('GPU Usage')}: ${results.gpuUsage}%)`);

    }
  });
}

async function initMidi() {
  EventsOn('midiError', (data: string) => {
    if (commonStore.platform === 'windows')
      toast('MIDI Error: ' + data, { type: 'error' });
  });
  EventsOn('midiPorts', (data: MidiPort[]) => {
    commonStore.setMidiPorts(data);
  });
  EventsOn('midiMessage', async (data: MidiMessage) => {
    await (await import('./pages/AudiotrackManager/AudiotrackEditor')).midiMessageHandler(data);
  });
  if (await FileExists('assets/sound-font/accordion/instrument.json')) {
    commonStore.setCompositionParams({
      ...commonStore.compositionParams,
      useLocalSoundFont: true
    });
  }
}
