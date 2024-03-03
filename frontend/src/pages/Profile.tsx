import React, { FC, useEffect, useRef } from 'react';
import { Page } from '../components/Page';
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Button,
  Dropdown,
  Input,
  Option,
  Switch
} from '@fluentui/react-components';
import { Labeled } from '../components/Labeled';
import commonStore from '../stores/commonStore';
import { observer } from 'mobx-react-lite';
import { useTranslation } from 'react-i18next';
import { checkUpdate, toastWithButton } from '../utils';
import { RestartApp } from '../../wailsjs/go/backend_golang/App';
import { Language, Languages } from '../types/settings';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { getServerRoot } from '../utils';

export const GeneralSettings: FC = observer(() => {
  const { t } = useTranslation();

  return <div className="flex flex-col gap-2">
    <Labeled label={t('Language')} flex spaceBetween content={
      <Dropdown style={{ minWidth: 0 }} listbox={{ style: { minWidth: 'fit-content' } }}
        value={Languages[commonStore.settings.language]}
        selectedOptions={[commonStore.settings.language]}
        onOptionSelect={(_, data) => {
          if (data.optionValue) {
            const lang = data.optionValue as Language;
            commonStore.setSettings({
              language: lang
            });
          }
        }}>
        {
          Object.entries(Languages).map(([langKey, desc]) =>
            <Option key={langKey} value={langKey}>{desc}</Option>)
        }
      </Dropdown>
    } />
    {
      commonStore.platform === 'windows' &&
      <Labeled label={t('DPI Scaling')} flex spaceBetween content={
        <Dropdown style={{ minWidth: 0 }} listbox={{ style: { minWidth: 'fit-content' } }}
          value={commonStore.settings.dpiScaling + '%'}
          selectedOptions={[commonStore.settings.dpiScaling.toString()]}
          onOptionSelect={(_, data) => {
            if (data.optionValue) {
              commonStore.setSettings({
                dpiScaling: Number(data.optionValue)
              });
              toastWithButton(t('Restart the app to apply DPI Scaling.'), t('Restart'), () => {
                RestartApp();
              }, {
                autoClose: 5000
              });
            }
          }}>
          {
            Array.from({ length: 7 }, (_, i) => (i + 2) * 25).map((v, i) =>
              <Option key={i} value={v.toString()}>{v + '%'}</Option>)
          }
        </Dropdown>
      } />
    }
    <Labeled label={t('Dark Mode')} flex spaceBetween content={
      <Switch checked={commonStore.settings.darkMode}
        onChange={(e, data) => {
          commonStore.setSettings({
            darkMode: data.checked
          });
        }} />
    } />
  </div>;
});

export const AdvancedGeneralSettings: FC = observer(() => {
  const { t } = useTranslation();

  let code: string = "";
  let phone_number: string = "";

  const currentConfig = commonStore.getCurrentModelConfig();
  const apiParams = currentConfig.apiParameters;
  const port = apiParams.apiPort;

  return <div className="flex flex-col gap-2">
    <Labeled label={t('Email')}
      content={
        <Input style={{ minWidth: 0 }} className="grow" placeholder={commonStore.settings.email} readOnly/>
      } />
    <Labeled label={t('User Name')}
      content={
        <Input style={{ minWidth: 0 }} className="grow" placeholder={commonStore.settings.username} readOnly/>
      } />
    <Labeled label={t('Phone Number')}
      content={
        <Input style={{ minWidth: 0 }} className="grow" placeholder={commonStore.settings.password} readOnly/>
      } />   
  </div>;
});

const Profile: FC = observer(() => {
  const { t } = useTranslation();
  const advancedHeaderRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (advancedHeaderRef.current)
      (advancedHeaderRef.current.firstElementChild as HTMLElement).style.padding = '0';
  }, []);

  return (
    <Page title={t('Profile')} content={
      <div className="flex flex-col gap-2 overflow-y-auto overflow-x-hidden p-1">
        {
          commonStore.platform === 'web' ?
            (
              <div className="flex flex-col gap-2">
                <GeneralSettings />
                <AdvancedGeneralSettings />
              </div>
            )
            :
            (
              <div className="flex flex-col gap-2">
                <GeneralSettings />
                <Labeled label={t('Automatic Updates Check')} flex spaceBetween content={
                  <Switch checked={commonStore.settings.autoUpdatesCheck}
                    onChange={(e, data) => {
                      commonStore.setSettings({
                        autoUpdatesCheck: data.checked
                      });
                      if (data.checked)
                        checkUpdate(true);
                    }} />
                } />
                {
                  commonStore.settings.language === 'zh' &&
                  <Labeled label={t('Use Gitee Updates Source')} flex spaceBetween content={
                    <Switch checked={commonStore.settings.giteeUpdatesSource}
                      onChange={(e, data) => {
                        commonStore.setSettings({
                          giteeUpdatesSource: data.checked
                        });
                      }} />
                  } />
                }
                {
                  commonStore.settings.language === 'zh' && commonStore.platform !== 'linux' &&
                  <Labeled label={t('Use Tsinghua Pip Mirrors')} flex spaceBetween content={
                    <Switch checked={commonStore.settings.cnMirror}
                      onChange={(e, data) => {
                        commonStore.setSettings({
                          cnMirror: data.checked
                        });
                      }} />
                  } />
                }
                <Labeled label={t('Allow external access to the API (service must be restarted)')} flex spaceBetween
                  content={
                    <Switch checked={commonStore.settings.host !== '127.0.0.1'}
                      onChange={(e, data) => {
                        commonStore.setSettings({
                          host: data.checked ? '0.0.0.0' : '127.0.0.1'
                        });
                      }} />
                  } />
                <Accordion collapsible openItems={!commonStore.advancedCollapsed && 'advanced'} onToggle={(e, data) => {
                  if (data.value === 'advanced')
                    commonStore.setAdvancedCollapsed(!commonStore.advancedCollapsed);
                }}>
                  <AccordionItem value="advanced">
                    <AccordionHeader ref={advancedHeaderRef} size="large">{t('Advanced')}</AccordionHeader>
                    <AccordionPanel>
                      <div className="flex flex-col gap-2 overflow-hidden">
                        {commonStore.platform !== 'darwin' &&
                          <Labeled label={t('Custom Models Path')}
                            content={
                              <Input className="grow" placeholder="./models"
                                value={commonStore.settings.customModelsPath}
                                onChange={(e, data) => {
                                  commonStore.setSettings({
                                    customModelsPath: data.value
                                  });
                                }} />
                            } />
                        }
                        <Labeled label={t('Custom Python Path')} // if set, will not use precompiled cuda kernel
                          content={
                            <Input className="grow" placeholder="./py310/python"
                              value={commonStore.settings.customPythonPath}
                              onChange={(e, data) => {
                                commonStore.setDepComplete(false);
                                commonStore.setSettings({
                                  customPythonPath: data.value
                                });
                              }} />
                          } />
                        <AdvancedGeneralSettings />
                      </div>
                    </AccordionPanel>
                  </AccordionItem>
                </Accordion>
              </div>
            )
        }
      </div>
    } />
  );
});

export default Profile;
