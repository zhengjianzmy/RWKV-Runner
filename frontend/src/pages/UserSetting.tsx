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


export const AdvancedGeneralSettings: FC = observer(() => {
  const { t } = useTranslation();

  let code: string = "";
  let phone_number: string = "";

  const currentConfig = commonStore.getCurrentModelConfig();
  const apiParams = currentConfig.apiParameters;
  const port = apiParams.apiPort;
  
  const saveAll = () => {
    fetchEventSource(
      getServerRoot(port, true) + '/v1/update_user',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${commonStore.settings.apiKey}`
        },
        body: JSON.stringify({
          username: commonStore.settings.username,
          email: commonStore.settings.email,
          password: commonStore.settings.password,
          phone_number: commonStore.settings.phoneNumber
        }),
        onmessage(e) {
          console.log(e.data)
          let data;
          try {
            data = JSON.parse(e.data);
          } catch (error) {
            console.debug('json error', error);
            return;
          }
          commonStore.setSettings({
            email: data.email
          });
          commonStore.setSettings({
            username: data.username
          });
          commonStore.setSettings({
            password: data.password
          });
        },
        async onopen(response) {
          console.log(response);
        },
        onclose() {
          window.alert(t('Update Success'));
          console.log('Connection closed');
        },
        onerror(err) {
          throw err;
        }
      });
    console.log('Connection closed');
  };

  const deleteUser = () => {
    fetchEventSource(
      getServerRoot(port, true) + '/v1/update_user',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${commonStore.settings.apiKey}`
        },
        body: JSON.stringify({
          username: "",
          email: "",
          password: "",
          phone_number: commonStore.settings.phoneNumber
        }),
        onmessage(e) {
          console.log(e.data)
          let data;
          try {
            data = JSON.parse(e.data);
          } catch (error) {
            console.debug('json error', error);
            return;
          }
        },
        async onopen(response) {
          console.log(response);
        },
        onclose() {
          window.alert(t('Update Success'));
          console.log('Connection closed');
        },
        onerror(err) {
          throw err;
        }
      });
    commonStore.setSettings({
      username: ""
    });
    commonStore.setSettings({
      password: ""
    });
    commonStore.setSettings({
      email: ""
    });
    commonStore.setSettings({
      phoneNumber: ""
    });
    commonStore.setSettings({
      code: ""
    });
    commonStore.setSettings({
      token: ""
    });
    commonStore.setSettings({
      oldCode: ""
    });
    commonStore.setSettings({
      uuid: ""
    });
    window.alert(t('Delete User'));
    console.log('Connection closed');
  };

  return <div className="flex flex-col gap-2">
    <Labeled label={t('Email')}
      content={
        <Input style={{ minWidth: 0 }} className="grow" placeholder="email" value={commonStore.settings.email}
          onChange={(e, data) => {
            commonStore.setSettings({
              email: data.value
            });
          }} />
      } />
    <Labeled label={t('User Name')}
      content={
          <Input style={{ minWidth: 0 }} className="grow" placeholder="username"
            value={commonStore.settings.username}
            onChange={(e, data) => {
              commonStore.setSettings({
                username: data.value
              });
            }} />
      } />
    <Labeled label={t('Password')}
      content={
          <Input style={{ minWidth: 0 }} className="grow" placeholder="password"
            value={commonStore.settings.password}
            onChange={(e, data) => {
              commonStore.setSettings({
                password: data.value
              });
            }} />
      } />
    <Labeled label={t('Save')}
      content={
          <Button appearance="primary" onClick={saveAll}>{t('Save')}</Button>
      } />
    <Labeled label={t('Delete User')}
      content={
          <Button appearance="primary" onClick={deleteUser}>{t('Delete')}</Button>
      } />
  </div>;
});

const UserSetting: FC = observer(() => {
  const { t } = useTranslation();
  const advancedHeaderRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (advancedHeaderRef.current)
      (advancedHeaderRef.current.firstElementChild as HTMLElement).style.padding = '0';
  }, []);

  return (
    <Page title={t('UserSetting')} content={
      <div className="flex flex-col gap-2 overflow-y-auto overflow-x-hidden p-1">
        {
          commonStore.platform === 'web' ?
            (
              <div className="flex flex-col gap-2">
                <AdvancedGeneralSettings />
              </div>
            )
            :
            (
              <div className="flex flex-col gap-2">
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

export default UserSetting;
