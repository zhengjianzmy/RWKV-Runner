import React, { FC, useEffect, useRef } from 'react';
import { Page } from '../components/Page';
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Button,
  Checkbox,
  Text,
  Textarea,
  Link,
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
  const sendCode = () => {
    fetchEventSource(
      getServerRoot(port, true) + '/v1/send_verification_code',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${commonStore.settings.apiKey}`
        },
        body: JSON.stringify({
          phone_number: commonStore.settings.phoneNumber
        }),
        onmessage(e) {
          // console.log(e)
          console.log(e.data)
          let data;
          try {
            data = JSON.parse(e.data);
          } catch (error) {
            console.debug('json error', error);
            return;
          }
          commonStore.setSettings({
            oldCode: data.code
          });
        },
        async onopen(response) {
          console.log(response);
        },
        onclose() {
          window.alert(t('Code Sent'));
          console.log('Connection closed');
        },
        onerror(err) {
          console.log('error');
          throw err;
        }
      });
    console.log('Connection closed');
  };

  const confirmCode = () => {
    if (commonStore.settings.privacyConfirmed !== true) {
      window.alert(t('Privacy Confirmed First'));
      return;
    }
    if (commonStore.settings.phoneNumber == "") {
      window.alert(t('Need Phone Number'));
      return;
    }
    if (commonStore.settings.code == "" || commonStore.settings.code !== commonStore.settings.oldCode) {
      console.log(code)
      window.alert(t('Verification Fail'));
      return;
    }
    fetchEventSource(
      getServerRoot(port, true) + '/v1/login/phone',
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
          // console.log("test");
          // console.log(e)
          console.log(e.data)
          let data;
          try {
            data = JSON.parse(e.data);
          } catch (error) {
            console.debug('json error', error);
            return;
          }
          commonStore.setSettings({
            uuid: data.uuid
          });
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
          window.alert(t('Login Success'));
          console.log('Connection closed');
        },
        onerror(err) {
          throw err;
        }
      });
    console.log('Connection closed');
  };

  const confirmPassword = () => {
    if (commonStore.settings.privacyConfirmed !== true) {
      window.alert(t('Privacy Confirmed First'));
      return;
    }
    if (commonStore.settings.phoneNumber == "") {
      window.alert(t('Need Phone Number'));
      return;
    }
    if (commonStore.settings.password == "") {
      window.alert(t('Need Password'));
      return;
    }
    fetchEventSource(
      getServerRoot(port, true) + '/v1/login/password',
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
          // console.log("test");
          // console.log(e)
          console.log(e.data)
          let data;
          try {
            data = JSON.parse(e.data);
          } catch (error) {
            console.debug('json error', error);
            return;
          }
          commonStore.setSettings({
            uuid: data.uuid
          });
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
          if (commonStore.settings.uuid != "") {
            window.alert(t('Login Success'));
            console.log('Connection closed');
          } else {
            window.alert(t('Login Fail'));
            console.log(commonStore.settings.password);
          }
        },
        onerror(err) {
          throw err;
        }
      });
    console.log('Connection closed');
  };

  const handleAgreement = (event: React.MouseEvent<HTMLAnchorElement, MouseEvent>) => {
    event.preventDefault();
    window.alert(t('Agreement Text'));
  };

  const handlePolicy = (event: React.MouseEvent<HTMLAnchorElement, MouseEvent>) => {
    event.preventDefault();
    window.alert(t('Private Policy Text'));
  };

  if (commonStore.settings.username != '' && commonStore.settings.phoneNumber != '') {
    return <div className="flex flex-col gap-2">
      <Labeled label={t('Already Login')}
        content={
          <div className="flex gap-2">
          </div>
        } />
    </div>;
  }

  return <div className="flex flex-col gap-2">
    <Labeled label={t('Phone Number')}
      content={
        <div className="flex gap-2">
          <Input style={{ minWidth: 0 }} className="grow" placeholder="phone number" value={commonStore.settings.phoneNumber}
            onChange={(e, data) => {
              commonStore.setSettings({
                phoneNumber: data.value
              });
            }} />
          <Button appearance="primary" onClick={sendCode}>{t('Send Code')}</Button>
        </div>
      } />
    <Labeled label={t('Code')}
      content={
        <div className="flex gap-2">
          <Input style={{ minWidth: 0 }} className="grow" placeholder="Code" value={commonStore.settings.code}
            onChange={(e, data) => {
              commonStore.setSettings({
                code: data.value
              });
            }} />
          <Button appearance="primary" onClick={confirmCode}>{t('Login')}</Button>
        </div>
      } />
    <Labeled label={t('Password')}
      content={
        <div className="flex gap-2">
          <Input type="password" className="grow" placeholder="Password" value={commonStore.settings.password}
            onChange={(e, data) => {
              commonStore.setSettings({
                password: data.value
              });
            }} />
          <Button appearance="primary" onClick={confirmPassword}>{t('LoginByPassword')}</Button>
        </div>
      } />
    <Labeled label={t('Register') }
      content={
          <Checkbox className="select-none"
                  size="large" label={(
                    <>
                      <Text>{t('You Confirmed')}</Text>
                      <Link href="http://www.luxitech.cn/agreement.html" target="_blank">{t('Agreement')}</Link>
                      <Text>{t('And')}</Text>
                      <Link href="http://www.luxitech.cn/privacy.html" target="_blank">{t('Private Policy')}</Link>
                    </>
                  )}
                  checked={commonStore.settings.privacyConfirmed}
                  onChange={(_, data) => {
                    commonStore.setSettings({
                      privacyConfirmed: data.checked as boolean
                    });
                  }} />
      } />
  </div>;
});

const Login: FC = observer(() => {
  const { t } = useTranslation();
  const advancedHeaderRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (advancedHeaderRef.current)
      (advancedHeaderRef.current.firstElementChild as HTMLElement).style.padding = '0';
  }, []);

  return (
    <Page title={t('Login')} content={
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

export default Login;
