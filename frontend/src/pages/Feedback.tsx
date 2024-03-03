import React, { FC, useEffect, useRef } from 'react';
import { Page } from '../components/Page';
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Checkbox,
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
      getServerRoot(port, true) + '/v1/insert_feedback',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${commonStore.settings.apiKey}`
        },
        body: JSON.stringify({
          id: commonStore.settings.uuid,
          notTruth: commonStore.settings.notTruth,
          timeout: commonStore.settings.timeout,
          notLogin: commonStore.settings.notLogin,
          notChat: commonStore.settings.notChat,
          bodyHit: commonStore.settings.bodyHit,
          policy: commonStore.settings.policy,
          sex: commonStore.settings.sex,
          notHealthy: commonStore.settings.notHealthy,
          others: commonStore.settings.others,
          description: commonStore.settings.description,
          contact: commonStore.settings.contact
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
    console.log('Connection closed');
  };

  return <div className="flex flex-col gap-2">
    <Labeled label={t('Not Truth')}
      content={
          <Checkbox className="select-none"
                  size="large" label={t('Not Truth')}
                  checked={commonStore.settings.notTruth}
                  onChange={(_, data) => {
                    commonStore.setSettings({
                      notTruth: data.checked as boolean
                    });
                  }} />
      } />
    <Labeled label={t('Timeout')}
      content={
          <Checkbox className="select-none"
                  size="large" label={t('Timeout')}
                  checked={commonStore.settings.timeout}
                  onChange={(_, data) => {
                    commonStore.setSettings({
                      timeout: data.checked as boolean
                    });
                  }} />
      } />
    <Labeled label={t('Can Not Login')}
      content={
          <Checkbox className="select-none"
                  size="large" label={t('Can Not Login')}
                  checked={commonStore.settings.notLogin}
                  onChange={(_, data) => {
                    commonStore.setSettings({
                      notLogin: data.checked as boolean
                    });
                  }} />
      } />
    <Labeled label={t('Can Not Chat')}
      content={
          <Checkbox className="select-none"
                  size="large" label={t('Can Not Chat')}
                  checked={commonStore.settings.notChat}
                  onChange={(_, data) => {
                    commonStore.setSettings({
                      notChat: data.checked as boolean
                    });
                  }} />
      } />
    <Labeled label={t('Body Hit')}
      content={
          <Checkbox className="select-none"
                  size="large" label={t('Body Hit')}
                  checked={commonStore.settings.bodyHit}
                  onChange={(_, data) => {
                    commonStore.setSettings({
                      bodyHit: data.checked as boolean
                    });
                  }} />
      } />
    <Labeled label={t('Policy')}
      content={
          <Checkbox className="select-none"
                  size="large" label={t('Policy')}
                  checked={commonStore.settings.policy}
                  onChange={(_, data) => {
                    commonStore.setSettings({
                      policy: data.checked as boolean
                    });
                  }} />
      } />
    <Labeled label={t('Sex')}
      content={
          <Checkbox className="select-none"
                  size="large" label={t('Sex')}
                  checked={commonStore.settings.sex}
                  onChange={(_, data) => {
                    commonStore.setSettings({
                      sex: data.checked as boolean
                    });
                  }} />
      } />
    <Labeled label={t('Not Healthy')}
      content={
          <Checkbox className="select-none"
                  size="large" label={t('Not Healthy')}
                  checked={commonStore.settings.notHealthy}
                  onChange={(_, data) => {
                    commonStore.setSettings({
                      notHealthy: data.checked as boolean
                    });
                  }} />
      } />
    <Labeled label={t('Others')}
      content={
          <Checkbox className="select-none"
                  size="large" label={t('Others')}
                  checked={commonStore.settings.others}
                  onChange={(_, data) => {
                    commonStore.setSettings({
                      others: data.checked as boolean
                    });
                  }} />
      } />
    <Labeled label={t('Description')}
      content={
          <Input style={{ minWidth: 0 }} className="grow" placeholder="description"
            value={commonStore.settings.description}
            onChange={(e, data) => {
              commonStore.setSettings({
                description: data.value
              });
            }} />
      } />
    <Labeled label={t('Contact')}
      content={
          <Input style={{ minWidth: 0 }} className="grow" placeholder="contact"
            value={commonStore.settings.contact}
            onChange={(e, data) => {
              commonStore.setSettings({
                contact: data.value
              });
            }} />
      } />
      <Labeled label={t('Save')}
        content={
            <Button appearance="primary" onClick={saveAll}>{t('Save')}</Button>
      } />
  </div>;
});

const Feedback: FC = observer(() => {
  const { t } = useTranslation();
  const advancedHeaderRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (advancedHeaderRef.current)
      (advancedHeaderRef.current.firstElementChild as HTMLElement).style.padding = '0';
  }, []);

  return (
    <Page title={t('Feedback')} content={
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

export default Feedback;
