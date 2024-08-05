import React, { FC, useState, useEffect, useRef } from 'react';
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

  const currentConfig = commonStore.getCurrentModelConfig();
  const apiParams = currentConfig.apiParameters;
  const port = apiParams.apiPort;

  const [file, setFile] = useState<any | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileUpload = async () => {
    if (!file) return;
    const fileArrayBuffer = await file.arrayBuffer();
    console.log(fileArrayBuffer);
    
    const filename = file.name;
    var contentType = 'application/pdf'
    let body = fileArrayBuffer;
    const formData = new FormData();
    formData.append('file', file);
    if (filename.endsWith(".docx")) {
      contentType = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
      body = file;
    } else if (filename.endsWith('.txt')) {
      contentType = 'text/plain';
    }
  
    // Make a post request to /process_pdf with the file
    fetch('/v1/file/process_pdf', {
        method: 'POST',
        headers: {
          'Content-Type': contentType,
          'Content-Length': fileArrayBuffer.byteLength
        },
        // body: JSON.stringify({
        //   body
        // }),
        body: body,
      }
    ).then(response => response.json())
    // Append the reply to #chat as a simple paragraph without any styling
    .then(data => {
      commonStore.setSettings({
        key: data.key
      });
    })
    .catch(error => {
      console.error(error);
    });
    //     onmessage(e) {
    //       // Append the reply to #chat as a simple paragraph without any styling
    //       console.log(e)
    //       let data;
    //       try {
    //         data = JSON.parse(e.data);
    //       } catch (error) {
    //         console.debug('json error', error);
    //         return;
    //       }
    //       commonStore.setSettings({
    //         key: data.key
    //       });
    //       console.log(data)
    //     },
    //     onerror(err) {
    //       console.error(err);
    //     }
    //  }
    // );
  
    const BASE_URL = "https://zhoupeng-paper.oss-us-east-1.aliyuncs.com/";
  
    fetch(BASE_URL + 'paper/' + filename, {
      method: "PUT",
      body: file
    });

    commonStore.setSettings({
      fileName: file.name
    });
    // window.alert(t('Upload File Success'));
    console.log('Connection closed');
  }

  return <div className="flex flex-col gap-2">


    <Labeled label={t('Upload File')}
      content={
        <div>
          <input type="file" accept=".pdf, .txt, .docx" ref={fileInputRef} style={{ display: 'none' }} onChange={handleFileChange}/>
          <Button onClick={handleButtonClick}>{t('Choose File')}</Button>

          {file && (
            <div>
              <p>Selected file: {file.name}</p>
              <Button onClick={handleFileUpload}>{t('Upload File')}</Button>
            </div>
          )}
        </div>
      } />
  </div>;

});

const UploadFile: FC = observer(() => {
  const { t } = useTranslation();
  const advancedHeaderRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (advancedHeaderRef.current)
      (advancedHeaderRef.current.firstElementChild as HTMLElement).style.padding = '0';
  }, []);

  return (
    <Page title={t('Upload File')} content={
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

export default UploadFile;
